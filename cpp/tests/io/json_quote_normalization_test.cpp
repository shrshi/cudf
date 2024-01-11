/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include <cudf/io/detail/json.hpp>
#include <cudf/io/json.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/default_stream.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

#include <string>

// Base test fixture for tests
struct JsonNormalizationTest : public cudf::test::BaseFixture {};

TEST_F(JsonNormalizationTest, ValidOutput)
{
  // RMM memory resource
  std::shared_ptr<rmm::mr::device_memory_resource> rsc =
    std::make_shared<rmm::mr::cuda_memory_resource>();

  // Test input
  std::string const input = R"({"A":'TEST"'})";
  rmm::device_uvector<char> device_input(input.size(), cudf::test::get_default_stream(), rsc.get());
  thrust::copy(input.begin(), input.end(), device_input.begin());
  auto device_input_span = cudf::device_span<std::byte>(
    reinterpret_cast<std::byte*>(device_input.data()), device_input.size());

  // Preprocessing FST
  auto device_fst_output_ptr = cudf::io::json::detail::normalize_quotes(
    device_input_span, cudf::test::get_default_stream(), rsc.get());

  // Initialize parsing options (reading json lines)
  auto device_fst_output_span = cudf::device_span<std::byte>(
    reinterpret_cast<std::byte*>(device_fst_output_ptr->data()), device_fst_output_ptr->size());
  cudf::io::json_reader_options input_options =
    cudf::io::json_reader_options::builder(cudf::io::source_info{device_fst_output_span});

  cudf::io::table_with_metadata processed_table =
    cudf::io::read_json(input_options, cudf::test::get_default_stream(), rsc.get());
}

CUDF_TEST_PROGRAM_MAIN()
