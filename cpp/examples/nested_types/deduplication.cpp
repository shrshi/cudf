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

#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/groupby.hpp>
#include <cudf/io/json.hpp>
#include <cudf/io/types.hpp>
#include <cudf/join.hpp>
#include <cudf/sorting.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table_view.hpp>

#include <chrono>
#include <iostream>
#include <string>

cudf::io::table_with_metadata read_json(std::string filepath)
{
  auto source_info = cudf::io::source_info(filepath);
  auto builder     = cudf::io::json_reader_options::builder(source_info).lines(true);
  auto options     = builder.build();
  return cudf::io::read_json(options);
}

void write_json(cudf::table_view tbl, cudf::io::table_metadata metadata, std::string filepath)
{
  // write the data for inspection
  auto sink_info = cudf::io::sink_info(filepath);
  auto builder2  = cudf::io::json_writer_options::builder(sink_info, tbl).lines(true);
  builder2.metadata(metadata);
  auto options2 = builder2.build();
  cudf::io::write_json(options2);
}

std::unique_ptr<cudf::table> count_aggregate(cudf::table_view tbl)
{
  // Get count for each key
  auto keys = cudf::table_view{{tbl.column(0)}};
  auto val  = tbl.column(0);
  cudf::groupby::groupby grpby_obj(keys);
  std::vector<cudf::groupby::aggregation_request> requests;
  requests.emplace_back(cudf::groupby::aggregation_request());
  auto agg = cudf::make_count_aggregation<cudf::groupby_aggregation>();
  requests[0].aggregations.push_back(std::move(agg));
  requests[0].values = val;
  auto agg_results   = grpby_obj.aggregate(requests);
  auto result_key    = std::move(agg_results.first);
  auto result_val    = std::move(agg_results.second[0].results[0]);
  std::vector<cudf::column_view> columns{result_key->get_column(0), *result_val};
  auto agg_v = cudf::table_view(columns);

  // Join on keys to get
  return std::make_unique<cudf::table>(agg_v);
}

std::unique_ptr<cudf::table> join_count(cudf::table_view left, cudf::table_view right)
{
  auto [left_indices, right_indices] =
    cudf::inner_join(cudf::table_view{{left.column(0)}}, cudf::table_view{{right.column(0)}});
  auto new_left  = cudf::gather(left, cudf::device_span<int const>{*left_indices});
  auto new_right = cudf::gather(right, cudf::device_span<int const>{*right_indices});

  auto left_cols  = new_left->release();
  auto right_cols = new_right->release();
  left_cols.push_back(std::move(right_cols[1]));

  return std::make_unique<cudf::table>(std::move(left_cols));
}

std::unique_ptr<cudf::table> sort_keys(cudf::table_view tbl)
{
  auto sort_order = cudf::sorted_order(cudf::table_view{{tbl.column(0)}});
  return cudf::gather(tbl, *sort_order);
}

/**
 * @brief Main for nested_types examples
 *
 * Command line parameters:
 * 1. JSON input file name/path (default: "example.json")
 * 2. JSON output file name/path (default: "output.json")
 *
 * The stdout includes the number of rows in the input and the output size in bytes.
 */
int main(int argc, char const** argv)
{
  std::string input_filepath;
  std::string output_filepath;
  if (argc < 2) {
    input_filepath  = "example.json";
    output_filepath = "output.json";
  } else if (argc == 3) {
    input_filepath  = argv[1];
    output_filepath = argv[2];
  } else {
    std::cout << "Either provide all command-line arguments, or none to use defaults" << std::endl;
    return 1;
  }

  // read input file
  auto [tbl, metadata] = read_json(input_filepath);

  auto st = std::chrono::steady_clock::now();

  auto count                               = count_aggregate(tbl->view());
  std::chrono::duration<double> count_time = std::chrono::steady_clock::now() - st;
  std::cout << "Wall time: " << count_time.count() << " seconds\n";
  auto combined                               = join_count(tbl->view(), count->view());
  std::chrono::duration<double> combined_time = std::chrono::steady_clock::now() - st;
  std::cout << "Wall time: " << combined_time.count() << " seconds\n";
  auto sorted                               = sort_keys(combined->view());
  std::chrono::duration<double> sorted_time = std::chrono::steady_clock::now() - st;
  std::cout << "Wall time: " << sorted_time.count() << " seconds\n";
  metadata.schema_info.emplace_back("count");

  std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - st;
  std::cout << "Wall time: " << elapsed.count() << " seconds\n";

  write_json(sorted->view(), metadata, output_filepath);

  return 0;
}
