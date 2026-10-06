/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>

#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/device_buffer.hpp>

#include <io/parquet/page_string_offsets.cuh>

#include <algorithm>
#include <cstring>
#include <limits>
#include <vector>

namespace pq = cudf::io::parquet::detail;

namespace {

constexpr uint32_t canary = 0xdeadbeef;

struct scan_case {
  std::vector<uint8_t> bytes;
  std::vector<uint32_t> offsets;
  int32_t value_count;
  bool active{true};
  uint32_t error{0};
};

void append_length(std::vector<uint8_t>& bytes, int32_t length)
{
  for (int i = 0; i < 4; ++i) {
    bytes.push_back(static_cast<uint32_t>(length) >> (i * 8));
  }
}

// Construct expectations from the strings themselves, independently of the GPU prefix walk.
scan_case make_case(std::vector<int32_t> const& lengths, uint32_t requested, int32_t value_count)
{
  scan_case result{{}, {}, value_count};
  for (auto const length : lengths) {
    result.offsets.push_back(result.bytes.size() + sizeof(int32_t));
    append_length(result.bytes, length);
    result.bytes.insert(result.bytes.end(), length, 0xa5);
  }
  result.offsets.push_back(result.bytes.size() + sizeof(int32_t));
  result.offsets.resize(requested + 1, result.offsets.back());
  return result;
}

struct device_case {
  pq::string_offset_scan_input input;
  int32_t value_count;
  bool active;
};

CUDF_KERNEL void scan_pages(cudf::device_span<device_case const> cases, uint32_t* errors)
{
  auto const warp =
    cooperative_groups::tiled_partition<32>(cooperative_groups::this_thread_block());
  auto const page = blockIdx.x * 4 + warp.meta_group_rank();
  if (page >= cases.size() || !cases[page].active) { return; }
  __shared__ uint32_t buffers[4][pq::string_offset_prefetch_words + 1];
  auto const& input = cases[page];
  pq::read_string_offsets(
    warp, input.input, input.value_count, buffers[warp.meta_group_rank()], errors + page);
}

void check_cases(std::vector<scan_case> const& cases)
{
  auto const stream = cudf::get_default_stream();
  auto const mr     = cudf::get_current_device_resource_ref();
  for (int alignment = 0; alignment < 4; ++alignment) {
    SCOPED_TRACE(alignment);
    std::vector<rmm::device_buffer> payloads;
    std::vector<device_case> descriptors;
    std::vector<uint32_t> expected;
    std::vector<size_t> starts;
    for (auto const& item : cases) {
      starts.push_back(expected.size() + 1);
      expected.push_back(canary);
      expected.insert(expected.end(), item.offsets.begin(), item.offsets.end());
      if (!item.active) { std::fill(expected.end() - item.offsets.size(), expected.end(), canary); }
      expected.push_back(canary);
    }
    std::vector<uint32_t> initial(expected.size(), canary);
    auto offsets =
      cudf::detail::make_device_uvector(cudf::host_span<uint32_t const>{initial}, stream, mr);
    for (size_t i = 0; i < cases.size(); ++i) {
      auto const& item = cases[i];
      std::vector<uint8_t> bytes(alignment, 0xff);
      bytes.insert(bytes.end(), item.bytes.begin(), item.bytes.end());
      // Keep an address for empty pages, which must not be dereferenced.
      if (bytes.empty()) { bytes.push_back(0xff); }
      payloads.emplace_back(bytes.data(), bytes.size(), stream, mr);
      stream.sync();
      descriptors.push_back({{static_cast<uint8_t const*>(payloads.back().data()) + alignment,
                              static_cast<int32_t>(item.bytes.size()),
                              static_cast<uint32_t>(item.offsets.size() - 1),
                              offsets.data() + starts[i]},
                             item.value_count,
                             item.active});
    }
    auto device_cases = cudf::detail::make_device_uvector(
      cudf::host_span<device_case const>{descriptors}, stream, mr);
    auto errors =
      cudf::detail::make_zeroed_device_uvector_async<uint32_t>(cases.size(), stream, mr);
    scan_pages<<<(cases.size() + 3) / 4, 128, 0, stream.get()>>>(device_cases, errors.data());
    CUDF_CUDA_TRY(cudaGetLastError());
    auto const actual = cudf::detail::make_std_vector(offsets, stream);
    EXPECT_EQ(expected, actual);
    auto const actual_errors = cudf::detail::make_std_vector(errors, stream);
    for (size_t i = 0; i < cases.size(); ++i) {
      EXPECT_EQ(cases[i].active ? cases[i].error : 0, actual_errors[i]) << "page " << i;
    }
  }
}

}  // namespace

struct ParquetStringOffsetsTest : cudf::test::BaseFixture {};

TEST_F(ParquetStringOffsetsTest, AlignmentsRefillsAndPartialBatches)
{
  std::vector<scan_case> cases;
  std::vector<int32_t> lengths(257);
  for (size_t i = 0; i < lengths.size(); ++i) {
    lengths[i] = (i * 13) % 97;
  }
  for (auto const requested : {0, 1, 31, 32, 33, 63, 64, 65, 256, 257, 270}) {
    cases.push_back(make_case(lengths, requested, lengths.size()));
  }
  cases.push_back(make_case(std::vector<int32_t>(257, 0), 260, 257));
  cases.push_back(make_case({1020, 0, 1, 1021, 2, 3, 1022, 4, 1023, 0}, 14, 1000));
  cases[1].active = false;  // Neighboring warps must still finish; final block is partly filled.
  check_cases(cases);
}

TEST_F(ParquetStringOffsetsTest, NullCountsAndPolicySwitch)
{
  std::vector<scan_case> cases;
  for (auto const length : {0, 1, 64, 65, 256, 4096}) {
    auto const lengths = std::vector<int32_t>(73, length);
    cases.push_back(make_case(lengths, 73, 73));
    cases.push_back(make_case(lengths, 10000, 10000));
  }
  cases.push_back(make_case({}, 128, 128));
  cases.push_back(make_case({}, 0, 0));
  // Partial reads after sparse refills exercise the buffered-to-direct handoff with a partial
  // output batch, and after a completed batch. All physical counts are deliberately unknown.
  std::vector<int32_t> skew(129, 1);
  std::fill(skew.begin() + 33, skew.begin() + 49, 256);
  for (auto const requested : {37, 42, 49, 65, 129, 150}) {
    cases.push_back(make_case(skew, requested, 10000));
  }
  check_cases(cases);
}

TEST_F(ParquetStringOffsetsTest, ExhaustionAndCorruptLengths)
{
  std::vector<scan_case> cases;
  for (auto const prefix_count : {0, 1, 31, 32, 33, 65}) {
    for (auto const string_length : {1, 256}) {
      auto const lengths = std::vector<int32_t>(prefix_count, string_length);
      for (auto const estimate : {1, 10000}) {
        for (int tail = 0; tail < 4; ++tail) {
          auto item = make_case(lengths, prefix_count + 5, estimate);
          item.bytes.insert(item.bytes.end(), tail, 0xff);
          cases.push_back(std::move(item));
        }
        for (auto const bad_length : {-1, 100, std::numeric_limits<int32_t>::max()}) {
          auto item = make_case(lengths, prefix_count + 5, estimate);
          append_length(item.bytes, bad_length);
          item.error = static_cast<uint32_t>(pq::decode_error::STRING_DATA_OVERRUN);
          cases.push_back(std::move(item));
        }
      }
    }
  }
  check_cases(cases);
}
