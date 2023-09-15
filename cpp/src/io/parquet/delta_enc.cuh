/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "parquet_gpu.hpp"

#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/integer_utils.hpp>

#include <cub/cub.cuh>

namespace cudf::io::parquet::gpu {

namespace delta {

inline __device__ void put_uleb128(uint8_t*& p, uleb128_t v)
{
  while (v > 0x7f) {
    *p++ = v | 0x80;
    v >>= 7;
  }
  *p++ = v;
}

inline __device__ uint8_t* put_zz128(uint8_t*& p, zigzag128_t v)
{
  zigzag128_t s = (v < 0);
  put_uleb128(p, (v ^ -s) * 2 + s);
}

// a block size of 128, with 4 mini-blocks of 32 values each fits nicely without consuming
// too much shared memory.
constexpr int block_size            = 128;
constexpr int num_mini_blocks       = 4;
constexpr int values_per_mini_block = block_size / num_mini_blocks;
constexpr int buffer_size           = 2 * block_size;

using block_reduce = cub::BlockReduce<zigzag128_t, block_size>;
using warp_reduce  = cub::WarpReduce<uleb128_t>;
using index_scan   = cub::BlockScan<size_type, block_size>;

constexpr int rolling_idx(int index) { return rolling_index<buffer_size>(index); }

// version of bit packer that can handle up to 64 bits values.
// T is the type to use for processing. if nbits <= 32 use uint32_t, otherwise unsigned long long
// (not uint64_t because of atomicOr's typing). allowing this to be selectable since there's a
// measurable impact to using the wider types.
template <typename scratch_type>
inline __device__ void bitpack_mini_block(
  uint8_t* dst, uleb128_t val, uint32_t count, uint8_t nbits, void* temp_space)
{
  using wide_type =
    std::conditional_t<std::is_same_v<scratch_type, unsigned long long>, __uint128_t, uint64_t>;
  using cudf::detail::warp_size;
  scratch_type constexpr mask = sizeof(scratch_type) * 8 - 1;
  auto constexpr div          = sizeof(scratch_type) * 8;

  auto const lane_id = threadIdx.x % warp_size;
  auto const warp_id = threadIdx.x / warp_size;

  auto const scratch = reinterpret_cast<scratch_type*>(temp_space) + warp_id * warp_size;

  // zero out scratch
  scratch[lane_id] = 0;
  __syncwarp();

  // TODO: see if there is any savings using special packing for easy bitwidths (1,2,4,8,16...)
  // like what's done for the RLE encoder.
  if (nbits == div) {
    if (lane_id < count) {
      for (int i = 0; i < sizeof(scratch_type); i++) {
        dst[lane_id * sizeof(scratch_type) + i] = val & 0xff;
        val >>= 8;
      }
    }
    __syncwarp();
    return;
  }

  if (lane_id <= count) {
    // shift symbol left by up to mask bits
    wide_type v2 = val;
    v2 <<= (lane_id * nbits) & mask;

    // Copy N bit word into two N/2 bit words while following C++ strict aliasing rules.
    scratch_type v1[2];
    memcpy(&v1, &v2, sizeof(wide_type));

    // Atomically write result to scratch
    if (v1[0]) { atomicOr(scratch + ((lane_id * nbits) / div), v1[0]); }
    if (v1[1]) { atomicOr(scratch + ((lane_id * nbits) / div) + 1, v1[1]); }
  }
  __syncwarp();

  // Copy scratch data to final destination
  auto const available_bytes = util::div_rounding_up_safe(count * nbits, 8U);
  auto const scratch_bytes   = reinterpret_cast<uint8_t const*>(scratch);

  for (uint32_t i = lane_id; i < available_bytes; i += warp_size) {
    dst[i] = scratch_bytes[i];
  }
  __syncwarp();
}

}  // namespace delta

// Object used to turn a stream of integers into a DELTA_BINARY_PACKED stream. This takes as input
// 128 values with validity at a time, saving them until there are enough values for a block
// to be written.
// T is the input data type (either zigzag128_t or uleb128_t)
template <typename T>
class DeltaBinaryPacker {
 private:
  uint8_t* _dst;                             // sink to dump encoded values to
  size_type _current_idx;                    // index of first value in buffer
  uint32_t _num_values;                      // total number of values to encode
  size_type _values_in_buffer;               // current number of values stored in _buffer
  T* _buffer;                                // buffer to store values to be encoded
  uint8_t _mb_bits[delta::num_mini_blocks];  // bitwidth for each mini-block

  // pointers to shared scratch memory for the warp and block scans/reduces
  delta::index_scan::TempStorage* _scan_tmp;
  delta::warp_reduce::TempStorage* _warp_tmp;
  delta::block_reduce::TempStorage* _block_tmp;

  void* _bitpack_tmp;  // pointer to shared scratch memory used in bitpacking

  // write the delta binary header. only call from thread 0
  inline __device__ void write_header(T first_value)
  {
    delta::put_uleb128(_dst, delta::block_size);
    delta::put_uleb128(_dst, delta::num_mini_blocks);
    delta::put_uleb128(_dst, _num_values);
    delta::put_zz128(_dst, first_value);
  }

  // write the block header. only call from thread 0
  inline __device__ void write_block_header(zigzag128_t block_min)
  {
    delta::put_zz128(_dst, block_min);
    memcpy(_dst, _mb_bits, 4);
    _dst += 4;
  }

 public:
  inline __device__ auto num_values() const { return _num_values; }

  // initialize the object. only call from thread 0
  inline __device__ void init(uint8_t* dest, uint32_t num_values, T* buffer, void* temp_storage)
  {
    _dst              = dest;
    _num_values       = num_values;
    _buffer           = buffer;
    _scan_tmp         = reinterpret_cast<delta::index_scan::TempStorage*>(temp_storage);
    _warp_tmp         = reinterpret_cast<delta::warp_reduce::TempStorage*>(temp_storage);
    _block_tmp        = reinterpret_cast<delta::block_reduce::TempStorage*>(temp_storage);
    _bitpack_tmp      = _buffer + delta::buffer_size;
    _current_idx      = 0;
    _values_in_buffer = 0;
  }

  // each thread calls this to add it's current value
  inline __device__ void add_value(T value, bool is_valid)
  {
    // figure out the correct position for the given value
    size_type const valid = is_valid;
    size_type pos;
    size_type num_valid;
    delta::index_scan(*_scan_tmp).ExclusiveSum(valid, pos, num_valid);

    if (is_valid) { _buffer[delta::rolling_idx(pos + _current_idx + _values_in_buffer)] = value; }
    __syncthreads();

    if (threadIdx.x == 0) {
      _values_in_buffer += num_valid;
      // if first pass write header
      if (_current_idx == 0) {
        write_header(_buffer[0]);
        _current_idx = 1;
        _values_in_buffer -= 1;
      }
    }
    __syncthreads();

    if (_values_in_buffer >= delta::block_size) { flush(); }
  }

  // called by each thread to flush data to the sink.
  inline __device__ uint8_t const* flush()
  {
    using cudf::detail::warp_size;
    __shared__ zigzag128_t block_min;

    int const t       = threadIdx.x;
    int const warp_id = t / warp_size;
    int const lane_id = t % warp_size;

    if (_values_in_buffer <= 0) { return _dst; }

    // calculate delta for this thread
    size_type const idx = _current_idx + t;
    zigzag128_t const delta =
      idx < _num_values ? _buffer[delta::rolling_idx(idx)] - _buffer[delta::rolling_idx(idx - 1)]
                        : std::numeric_limits<zigzag128_t>::max();

    // find min delta for the block
    auto const min_delta = delta::block_reduce(*_block_tmp).Reduce(delta, cub::Min());

    if (t == 0) { block_min = min_delta; }
    __syncthreads();

    // compute frame of reference for the block
    uleb128_t const norm_delta = idx < _num_values ? delta - block_min : 0;

    // get max normalized delta for each warp, and use that to determine how many bits to use
    // for the bitpacking of this warp
    zigzag128_t const warp_max =
      delta::warp_reduce(_warp_tmp[warp_id]).Reduce(norm_delta, cub::Max());
    __syncthreads();

    if (lane_id == 0) { _mb_bits[warp_id] = sizeof(zigzag128_t) * 8 - __clzll(warp_max); }
    __syncthreads();

    // write block header
    if (t == 0) { write_block_header(block_min); }
    __syncthreads();

    // now each warp encodes it's data...can calculate starting offset with _mb_bits
    uint8_t* mb_ptr = _dst;
    switch (warp_id) {
      case 3: mb_ptr += _mb_bits[2] * delta::values_per_mini_block / 8; [[fallthrough]];
      case 2: mb_ptr += _mb_bits[1] * delta::values_per_mini_block / 8; [[fallthrough]];
      case 1: mb_ptr += _mb_bits[0] * delta::values_per_mini_block / 8;
    }

    // encoding happens here....will have to update pack literals to deal with larger numbers
    auto const warp_idx = _current_idx + warp_id * delta::values_per_mini_block;
    if (warp_idx < _num_values) {
      auto const num_enc = min(delta::values_per_mini_block, _num_values - warp_idx);
      if (_mb_bits[warp_id] > 32) {
        delta::bitpack_mini_block<unsigned long long>(
          mb_ptr, norm_delta, num_enc, _mb_bits[warp_id], _bitpack_tmp);
      } else {
        delta::bitpack_mini_block<uint32_t>(
          mb_ptr, norm_delta, num_enc, _mb_bits[warp_id], _bitpack_tmp);
      }
    }
    __syncthreads();

    // last warp updates global delta ptr
    if (warp_id == delta::num_mini_blocks - 1 && lane_id == 0) {
      _dst              = mb_ptr + _mb_bits[warp_id] * delta::values_per_mini_block / 8;
      _current_idx      = min(warp_idx + delta::values_per_mini_block, _num_values);
      _values_in_buffer = max(_values_in_buffer - delta::block_size, 0U);
    }
    __syncthreads();

    return _dst;
  }
};

}  // namespace cudf::io::parquet::gpu