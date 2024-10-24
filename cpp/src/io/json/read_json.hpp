/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

#include "../comp/io_uncomp.hpp"

#include <cudf/io/datasource.hpp>
#include <cudf/io/json.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

#include <memory>

namespace CUDF_EXPORT cudf {
namespace io::json::detail {

// Some magic numbers
constexpr int num_subchunks               = 10;  // per chunk_size
constexpr size_t min_subchunk_size        = 10000;
constexpr int estimated_compression_ratio = 4;
constexpr int max_subchunks_prealloced    = 3;

/**
 * @brief Read from array of data sources into RMM buffer. The size of the returned device span
          can be larger than the number of bytes requested from the list of sources when
          the range to be read spans across multiple sources. This is due to the delimiter
          characters inserted after the end of each accessed source.
 *
 * @param buffer Device span buffer to which data is read
 * @param sources Array of data sources
 * @param compression Compression format of source
 * @param range_offset Number of bytes to skip from source start
 * @param range_size Number of bytes to read from source
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @returns A subspan of the input device span containing data read
 */
device_span<char> ingest_raw_input(device_span<char> buffer,
                                   host_span<std::unique_ptr<datasource>> sources,
                                   compression_type compression,
                                   size_t range_offset,
                                   size_t range_size,
                                   rmm::cuda_stream_view stream);

/**
 * @brief Reads and returns the entire data set in batches.
 *
 * @param sources Input `datasource` objects to read the dataset from
 * @param reader_opts Settings for controlling reading behavior
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource to use for device memory allocation
 *
 * @return cudf::table object that contains the array of cudf::column.
 */
table_with_metadata read_json(host_span<std::unique_ptr<datasource>> sources,
                              json_reader_options const& reader_opts,
                              rmm::cuda_stream_view stream,
                              rmm::device_async_resource_ref mr);

class compressed_host_buffer_source final : public datasource {
 public:
  explicit compressed_host_buffer_source(cudf::host_span<std::uint8_t const> ch_buffer,
                                         compression_type comptype)
    : _ch_buffer{ch_buffer}, _comptype{comptype}
  {
    if (comptype == compression_type::GZIP || comptype == compression_type::ZIP ||
        comptype == compression_type::SNAPPY) {
      _decompressed_ch_buffer_size = estimate_uncompressed_size(_comptype, _ch_buffer);
      _decompressed_buffer.resize(0);
    } else {
      _decompressed_buffer         = decompress(_comptype, _ch_buffer);
      _decompressed_ch_buffer_size = _decompressed_buffer.size();
    }
  }

  size_t host_read(size_t offset, size_t size, uint8_t* dst) override
  {
    auto decompressed_hbuf = decompress(_comptype, _ch_buffer);
    auto const count       = std::min(size, decompressed_hbuf.size() - offset);
    std::memcpy(dst, decompressed_hbuf.data() + offset, count);
    return count;
  }

  std::unique_ptr<buffer> host_read(size_t offset, size_t size) override
  {
    if (_decompressed_buffer.empty()) {
      auto decompressed_hbuf = decompress(_comptype, _ch_buffer);
      auto const count       = std::min(size, decompressed_hbuf.size() - offset);
      bool partial_read      = offset + count < decompressed_hbuf.size();
      if (!partial_read)
        return std::make_unique<owning_buffer<std::vector<uint8_t>>>(
          std::move(decompressed_hbuf), decompressed_hbuf.data() + offset, count);
      _decompressed_buffer = std::move(decompressed_hbuf);
    }
    auto const count = std::min(size, _decompressed_buffer.size() - offset);
    return std::make_unique<non_owning_buffer>(_decompressed_buffer.data() + offset, count);
  }

  [[nodiscard]] bool supports_device_read() const override { return false; }

  [[nodiscard]] size_t size() const override { return _decompressed_ch_buffer_size; }

 private:
  cudf::host_span<std::uint8_t const> _ch_buffer;  ///< A non-owning view of the existing host data
  compression_type _comptype;
  size_t _decompressed_ch_buffer_size;
  std::vector<std::uint8_t> _decompressed_buffer;
};

}  // namespace io::json::detail
}  // namespace CUDF_EXPORT cudf
