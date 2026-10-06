/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "parquet_gpu.hpp"

#include <cudf/detail/utilities/cuda.cuh>

#include <cooperative_groups.h>
#include <cuda/std/algorithm>
#include <cuda/std/cstdint>
#include <cuda/std/cstring>

namespace cudf::io::parquet::detail {

using string_offset_warp = cooperative_groups::thread_block_tile<cudf::detail::warp_size>;

// Keep the dispatch threshold independent of the prefetch allocation.
constexpr int string_offset_prefetch_size  = 1024;
constexpr int max_buffered_string_length   = 64;
constexpr int max_buffered_string_stride   = max_buffered_string_length + sizeof(int32_t);
constexpr int string_offset_prefetch_words = string_offset_prefetch_size / sizeof(uint32_t);

struct string_offset_scan_input {
  uint8_t const* data;
  int32_t bytes;
  uint32_t num_values;
  uint32_t* offsets;
};

/**
 * @brief Scan the remaining prefixes in one lane and cooperatively fill the terminal offsets.
 *
 * The initial cursor may follow a buffered scan. A short final prefix is natural exhaustion;
 * negative lengths and lengths extending beyond the page are corruption.
 */
inline __device__ void read_string_offsets_direct(string_offset_warp const& warp,
                                                  string_offset_scan_input const& input,
                                                  uint32_t pos,
                                                  uint32_t next,
                                                  kernel_error::pointer error_code)
{
  if (warp.thread_rank() == 0) {
    for (; pos < input.num_values; ++pos) {
      auto const string_offset = next + sizeof(int32_t);
      if (static_cast<int64_t>(string_offset) > input.bytes) { break; }
      int32_t len;
      cuda::std::memcpy(&len, input.data + next, sizeof(len));
      if (len < 0 || static_cast<int64_t>(string_offset) + len > input.bytes) {
        set_error(static_cast<kernel_error::value_type>(decode_error::STRING_DATA_OVERRUN),
                  error_code);
        break;
      }
      input.offsets[pos] = string_offset;
      next               = string_offset + len;
    }
  }
  pos  = warp.shfl(pos, 0);
  next = warp.shfl(next, 0);
  // The extra offset is intentionally four bytes beyond the consumed payload.
  for (size_t i = size_t{pos} + warp.thread_rank(); i <= input.num_values; i += warp.size()) {
    input.offsets[i] = next + sizeof(int32_t);
  }
}

/**
 * @brief Refill using consecutive aligned words across lanes, guarding both page edges.
 *
 * Align the source, not just the destination: otherwise every interior word needs byte assembly.
 * The first logical base may be negative for an unaligned page; never read those preceding bytes.
 * Callers synchronize the warp before reusing the buffer and after this refill.
 */
inline __device__ void prefetch_string_data(string_offset_warp const& warp,
                                            string_offset_scan_input const& input,
                                            uint32_t next,
                                            uint32_t* buffer,
                                            int32_t& base,
                                            int32_t& end)
{
  auto const misalignment = (reinterpret_cast<uintptr_t>(input.data) + next) & 3;
  base                    = static_cast<int32_t>(next) - static_cast<int32_t>(misalignment);
  end                     = static_cast<int32_t>(
    cuda::std::min<int64_t>(input.bytes, int64_t{base} + string_offset_prefetch_size));
  int const words = (end - base + 3) / sizeof(uint32_t);
  for (int w = warp.thread_rank(); w < words; w += warp.size()) {
    auto const offset = int64_t{base} + int64_t{sizeof(uint32_t)} * w;
    uint32_t word     = 0;
    if (offset >= 0 && offset + sizeof(uint32_t) <= input.bytes) {
      word = *reinterpret_cast<uint32_t const*>(input.data + offset);
    } else {
#pragma unroll
      for (int byte = 0; byte < sizeof(uint32_t); ++byte) {
        auto const index = offset + byte;
        if (index >= 0 && index < input.bytes) {
          word |= static_cast<uint32_t>(input.data[index]) << (8 * byte);
        }
      }
    }
    buffer[w] = word;
  }
  // An aligned prefix does not use the upper word, but the funnel load still reads it.
  if (warp.thread_rank() == 0) { buffer[words] = 0; }
}

/**
 * @brief Cooperatively scan lengths, batching offset stores and adapting to sparse prefixes.
 *
 * All lanes walk the same dependent prefix chain. Each lane retains one output offset, so every
 * group of 32 validated lengths produces a coalesced store. At refill boundaries, measure the
 * actual bytes advanced per physical value. Two consecutive sparse windows switch the remaining
 * scan to direct loads, even if nulls made the initial logical-count estimate too small. A single
 * unusually long string does not immediately change policy. No prefix is reparsed on switching.
 */
inline __device__ void read_string_offsets_buffered(string_offset_warp const& warp,
                                                    string_offset_scan_input const& input,
                                                    uint32_t* buffer,
                                                    kernel_error::pointer error_code)
{
  uint32_t next           = 0;
  uint32_t pos            = 0;
  uint32_t pending_offset = 0;
  uint32_t refill_pos     = 0;
  uint32_t refill_next    = 0;
  int32_t base            = 0;
  int32_t end             = 0;
  bool previous_sparse    = false;
  bool use_direct         = false;

  for (; pos < input.num_values; ++pos) {
    auto const string_offset = next + sizeof(int32_t);
    if (static_cast<int64_t>(string_offset) > input.bytes) { break; }
    if (string_offset > static_cast<uint32_t>(end)) {
      if (pos != refill_pos) {
        bool const sparse =
          uint64_t{next - refill_next} > uint64_t{pos - refill_pos} * max_buffered_string_stride;
        if (sparse && previous_sparse) {
          use_direct = true;
          break;
        }
        previous_sparse = sparse;
      }
      refill_pos  = pos;
      refill_next = next;
      warp.sync();
      prefetch_string_data(warp, input, next, buffer, base, end);
      warp.sync();
    }
    auto const index = static_cast<int32_t>(next) - base;
    auto const word  = index / sizeof(uint32_t);
    auto const shift = (index & 3) * 8;
    auto const len   = static_cast<int32_t>(__funnelshift_r(buffer[word], buffer[word + 1], shift));
    if (len < 0 || static_cast<int64_t>(string_offset) + len > input.bytes) {
      if (warp.thread_rank() == 0) {
        set_error(static_cast<kernel_error::value_type>(decode_error::STRING_DATA_OVERRUN),
                  error_code);
      }
      break;
    }
    next = string_offset + len;
    if (pos % warp.size() == warp.thread_rank()) { pending_offset = string_offset; }
    if (pos % warp.size() == warp.size() - 1) {
      input.offsets[pos - (warp.size() - 1) + warp.thread_rank()] = pending_offset;
    }
  }

  // Flush only validated offsets, including partial batches before a switch or corruption.
  auto const remainder = pos % warp.size();
  if (warp.thread_rank() < remainder) {
    input.offsets[pos - remainder + warp.thread_rank()] = pending_offset;
  }
  if (use_direct) {
    read_string_offsets_direct(warp, input, pos, next, error_code);
  } else {
    for (size_t i = size_t{pos} + warp.thread_rank(); i <= input.num_values; i += warp.size()) {
      input.offsets[i] = next + sizeof(int32_t);
    }
  }
}

/**
 * @brief Select an initial scan policy without division or unsigned subtraction.
 *
 * `value_count` is a full-page physical count when known, otherwise a logical upper bound. The
 * buffered scanner corrects the latter estimate using observed physical prefixes at refills.
 */
inline __device__ void read_string_offsets(string_offset_warp const& warp,
                                           string_offset_scan_input const& input,
                                           int32_t value_count,
                                           uint32_t* buffer,
                                           kernel_error::pointer error_code)
{
  if (input.bytes > 0 && value_count > 0 &&
      int64_t{input.bytes} <= int64_t{value_count} * max_buffered_string_stride) {
    read_string_offsets_buffered(warp, input, buffer, error_code);
  } else {
    read_string_offsets_direct(warp, input, 0, 0, error_code);
  }
}

}  // namespace cudf::io::parquet::detail
