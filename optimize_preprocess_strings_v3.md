# String-offset preprocessing implementation on ykiran-string-v3

This implements the first three follow-up ideas from `optimize_preprocess_strings_findings.md`:
aligned coalesced prefetch, concurrent page setup within a four-warp block, and improved scan
selection. Performance of this implementation has not been measured. Earlier experiment timings
in that report do not apply automatically to this implementation.

## Implementation

`cpp/src/io/parquet/page_string_offsets.cuh` contains the shared scanner used by production and
the focused CUDA tests. A warp walks one PLAIN length-prefix chain. Refills start at an absolute
four-byte boundary and copy consecutive 32-bit words across lanes into a 1 KiB shared buffer.
Only partial words at the page edges need byte assembly. Unaligned prefixes are reconstructed
from two shared words with a funnel shift. An extra initialized word makes the upper load safe
at the buffer boundary. Validated offsets are held one per lane and written in batches of 32;
partial batches are flushed before exhaustion, corruption, or a policy switch.

`preprocess_string_offsets_kernel` in `page_string_decode.cu` assigns four independent pages to
four warps. Unlike the previous experiment's sequential setup loop, each warp owns a separate
page state and calls `setup_local_page_info` concurrently. The setup helper accepts a cooperative
group and uses that group's rank, size, and synchronization; existing callers default to the
entire block. This preserves the existing level-section and row-bound handling instead of
duplicating it. The offset state no longer includes unused decode-progress counters. Setup
errors are checked before accessing the value stream.

The launcher uses one warp per block for fewer than four pages, otherwise four. This only avoids
an incompletely occupied grouped launch for the smallest input; it is not a measured performance
crossover or a solution to every regression observed in the previous experiments.

## Scan policy

The buffer capacity and dispatch threshold are independent: 1 KiB and 64 character bytes per
string, respectively. The initial comparison is

```
value_bytes <= full_page_value_count * (64 + 4)
```

using widened arithmetic. This removes division, unsigned subtraction, and the old wraparound on
pages with fewer than four bytes per logical value. Empty/all-null payloads fill terminal offsets
without dereferencing the payload.

When complete page-index metadata is available, the persisted full-page character byte count
provides the number of physical PLAIN strings:

```
physical_values = (value_bytes - str_bytes_from_index) / 4
```

The byte difference must be nonnegative and divisible by four. The policy does not use
`num_valids` or `num_nulls`, which may have been recomputed for a cropped range or prior chunked
read. Without complete index information, it initially uses the logical count as an upper bound.
For example, 256-byte strings with 75% nulls looked like 61-byte strings under the old estimate.
With complete index metadata, they now select direct scanning immediately; without it, the
observed physical strides allow a switch after two sparse windows.

At each refill boundary, the buffered scanner measures bytes advanced per actual parsed value.
Two consecutive windows averaging more than 68 bytes per prefix-plus-payload switch the remaining
scan to direct global prefix loads. Pending offsets are flushed, and the direct scanner resumes
at the exact current value and byte cursor. It does not reparse the page, allocate memory, or
launch another kernel. The switch is one-way and the two-window rule is an unbenchmarked heuristic:
it limits reactions to a single long string, but can still choose poorly if the distribution changes
later in the page. It is not a general classifier for every skewed distribution.

## Correctness and validation

Preserved scanner behavior includes payload-start offsets, the terminal `consumed_end + 4`
sentinel, natural exhaustion on fewer than four remaining bytes, and `STRING_DATA_OVERRUN` for
negative or overlong lengths. Requested row ranges still scan from the beginning of the page;
list continuation pages retain the existing leaf batch-size rule. Page masks, encoding filters,
and out-of-range warps can return independently because the kernel has no block-wide barriers.

The new `ParquetStringOffsetsTest` suite has 804 page/alignment cases across three tests. It
covers all four input alignments, refill boundaries, empty strings/payloads, unknown physical
counts, partial requests, partial output batches, policy handoffs, masked warps, short tails,
negative lengths, truncated bodies, and near-INT_MAX lengths. Expected offsets are constructed
from the input strings; per-page output canaries and individual error masks are checked.

The added `ParquetReaderTest.PlainStringOffsetsAcrossPages` covers V1/V2 PLAIN pages, with and
without column statistics, nullable long strings, skewed strings, all-null columns, and cropped
reads. Existing reader tests provide nested/list continuation and other encoding coverage.

Build environment: `cudf` conda environment, CUDA compiler targeting `sm_120a`. Initial compiler
resource output reports 40 registers/thread, zero spills, and zero block barriers for both
launcher specializations; shared memory is 5,904 bytes for four warps and 1,540 bytes for one.
These are compiler resource measurements, not runtime performance results.

The scanner tests passed directly and under Compute Sanitizer memcheck, racecheck, and synccheck
with zero reported errors or hazards. Full library build and reader validation are in progress.

Reproduce after building:

```bash
source /home/ykiran/miniconda3/etc/profile.d/conda.sh
conda activate cudf
cmake --build cpp/build --target PARQUET_TEST --parallel 8
cpp/build/gtests/PARQUET_TEST \
  --gtest_filter='ParquetStringOffsetsTest.*:ParquetReaderTest.PlainStringOffsetsAcrossPages'
compute-sanitizer --tool memcheck --error-exitcode 86 \
  cpp/build/gtests/PARQUET_TEST \
  --gtest_filter='ParquetStringOffsetsTest.*:ParquetReaderTest.PlainStringOffsetsAcrossPages'
```
