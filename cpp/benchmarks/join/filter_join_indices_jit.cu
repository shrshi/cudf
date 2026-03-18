/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "generate_input_tables.hpp"
#include "join_common.hpp"

#include <cudf/ast/expressions.hpp>
#include <cudf/join/hash_join.hpp>
#include <cudf/join/join.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/span.hpp>

#include <nvbench/nvbench.cuh>

/**
 * @brief Benchmark comparing AST interpreter vs JIT compilation for filter_join_indices.
 *
 * Both methods use the same AST expression, allowing fair comparison of:
 * - AST: Runtime tree traversal and interpretation
 * - JIT: One-time compilation to fused CUDA kernel
 *
 * The ast_levels parameter controls expression tree depth, measuring how each
 * approach scales with predicate complexity.
 */
template <typename JoinFunc>
void filter_join_indices_benchmark(nvbench::state& state,
                                   JoinFunc join_func,
                                   cudf::join_kind join_kind)
{
  auto const build_size = static_cast<cudf::size_type>(state.get_int64("build_size"));
  auto const probe_size = static_cast<cudf::size_type>(state.get_int64("probe_size"));
  auto const ast_levels = static_cast<cudf::size_type>(state.get_int64("ast_levels"));
  auto const method     = state.get_string("method");

  // Generate build (right) and probe (left) tables
  // 1 key column + 2 payload columns = 3 columns total
  std::vector<cudf::type_id> key_types{cudf::type_id::INT32};

  auto [build_table, probe_table] =
    generate_input_tables<false>(key_types, build_size, probe_size, 2, 1, 0.3);

  // Perform hash join on key column (column 0) to get indices
  auto probe_keys = probe_table->view().select({0});
  auto build_keys = build_table->view().select({0});

  cudf::hash_join hash_joiner(build_keys, cudf::null_equality::EQUAL);
  auto [left_indices, right_indices] = join_func(hash_joiner, probe_keys);

  cudf::device_span<cudf::size_type const> left_span{left_indices->data(), left_indices->size()};
  cudf::device_span<cudf::size_type const> right_span{right_indices->data(), right_indices->size()};

  // Build AST expression tree
  cudf::ast::tree tree;
  create_complex_ast_expression(tree, ast_levels);

  auto const join_input_size = estimate_size(build_keys) + estimate_size(probe_keys);
  auto const filter_input_size =
    join_input_size + (left_indices->size() + right_indices->size()) * sizeof(cudf::size_type);
  state.add_element_count(filter_input_size, "filter_input_size");
  state.template add_global_memory_reads<nvbench::int8_t>(filter_input_size);

  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));

  if (method == "AST") {
    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
      auto result = cudf::filter_join_indices(
        probe_table->view(), build_table->view(), left_span, right_span, tree.back(), join_kind);
    });
  } else {
    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
      auto result = cudf::filter_join_indices_jit(
        probe_table->view(), build_table->view(), left_span, right_span, tree.back(), join_kind);
    });
  }
  set_throughputs(state);
}

void filter_join_indices_jit_inner_join(nvbench::state& state)
{
  filter_join_indices_benchmark(
    state,
    [](cudf::hash_join& joiner, cudf::table_view probe_keys) {
      return joiner.inner_join(probe_keys);
    },
    cudf::join_kind::INNER_JOIN);
}

void filter_join_indices_jit_left_join(nvbench::state& state)
{
  filter_join_indices_benchmark(
    state,
    [](cudf::hash_join& joiner, cudf::table_view probe_keys) {
      return joiner.left_join(probe_keys);
    },
    cudf::join_kind::LEFT_JOIN);
}

NVBENCH_BENCH(filter_join_indices_jit_inner_join)
  .set_name("filter_join_indices_jit_inner")
  .add_string_axis("method", {"AST", "JIT"})
  .add_int64_axis("ast_levels", {1, 5, 10})
  .add_int64_power_of_two_axis("build_size", nvbench::range(12, 24, 2))
  .add_int64_power_of_two_axis("probe_size", nvbench::range(12, 24, 2));

NVBENCH_BENCH(filter_join_indices_jit_left_join)
  .set_name("filter_join_indices_jit_left")
  .add_string_axis("method", {"AST", "JIT"})
  .add_int64_axis("ast_levels", {1, 5, 10})
  .add_int64_power_of_two_axis("build_size", nvbench::range(12, 24, 2))
  .add_int64_power_of_two_axis("probe_size", nvbench::range(12, 24, 2));
