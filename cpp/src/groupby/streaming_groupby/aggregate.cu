/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "common.cuh"
#include "groupby/common/utils.hpp"
#include "warp_aggregate.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/iterator>
#include <cuda/stream>
#include <thrust/copy.h>
#include <thrust/fill.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <mutex>
#include <string>

namespace cudf::groupby {

void streaming_groupby::impl::do_aggregate(table_view const& data, cuda::stream_ref stream)
{
  ensure_not_invalidated();

  auto const batch_size = data.num_rows();
  if (batch_size == 0) { return; }

  CUDF_EXPECTS(batch_size <= _max_distinct_keys,
               "Batch size (" + std::to_string(batch_size) + ") exceeds max_distinct_keys (" +
                 std::to_string(_max_distinct_keys) + ").",
               std::invalid_argument);

  CUDF_EXPECTS(static_cast<int64_t>(_max_distinct_keys) + static_cast<int64_t>(batch_size) <=
                 static_cast<int64_t>(std::numeric_limits<size_type>::max()),
               "Transient key encoding (max_distinct_keys + batch_size) would overflow size_type.",
               std::invalid_argument);

  // Keep host bookkeeping locked until every result use is enqueued and registered.
  std::lock_guard const lock{_insert_mutex};
  ensure_not_invalidated();
  bool insertion_started = false;
  try {
    if (!_initialized) { initialize(data, stream); }
    // An initial tiny batch must not permanently limit later batches to one partial. Increasing
    // the budget only after doubling bounds how often stable groups require a reshape.
    auto const batch_budget = std::max<size_type>(1, batch_size / partial_aggregation_block_size);
    if (_partial_budget == 0 || static_cast<int64_t>(batch_budget) >= 2LL * _partial_budget) {
      _partial_budget = batch_budget;
    }
    auto const batch_keys = data.select(_key_indices);
    update_nullable_state(batch_keys);
    if (!_key_set) { create_key_set(stream); }
    _insert_done.wait(stream);
    insertion_started  = true;
    auto const targets = probe_and_insert(batch_keys, true, stream);
    _insert_done.record(stream);
    _results_read_done.wait(stream);

    compute_warp_aggregates(data.select(_value_col_indices),
                            _agg_kinds,
                            {targets.data(), targets.size()},
                            _agg_results->mutable_view(),
                            stream,
                            cudf::get_current_device_resource_ref());
    record_aggregate_completion(stream);
  } catch (...) {
    if (insertion_started) { _invalidated = true; }
    // Do not let partially enqueued writes escape tracking before a subsequent resize/read.
    static_cast<void>(cudaStreamSynchronize(stream.get()));
    throw;
  }
}

namespace {
struct is_unmatched_fn {
  size_type const* targets;
  bitmask_type const* row_bitmask;

  __device__ bool operator()(size_type row) const
  {
    return targets[row] == cudf::detail::CUDF_SIZE_TYPE_SENTINEL &&
           (!row_bitmask || cudf::bit_is_set(row_bitmask, row));
  }
};
}  // namespace

std::unique_ptr<column> streaming_groupby::impl::do_aggregate_existing(
  table_view const& data, cuda::stream_ref stream, rmm::device_async_resource_ref mr)
{
  std::lock_guard const lock{_insert_mutex};
  ensure_not_invalidated();
  auto const size = data.num_rows();
  if (size == 0) { return make_empty_column(data_type{type_id::INT32}); }
  CUDF_EXPECTS(static_cast<int64_t>(_max_distinct_keys) + size <=
                 static_cast<int64_t>(std::numeric_limits<size_type>::max()),
               "Lookup key encoding (max_distinct_keys + batch_size) would overflow size_type.",
               std::invalid_argument);

  try {
    auto const temp_mr = cudf::get_current_device_resource_ref();
    auto const keys    = data.select(_key_indices);
    update_nullable_state(keys);
    auto [mask, row_bitmask] =
      _has_nullable_keys && _null_handling == null_policy::EXCLUDE
        ? detail::compute_row_bitmask(keys, stream)
        : std::pair<cuda::device_buffer<std::byte>, bitmask_type const*>{
            cudf::create_null_mask(0, mask_state::UNALLOCATED, stream), nullptr};
    cuda::device_buffer<size_type> targets(stream, temp_mr, size, cuda::no_init);
    _insert_done.wait(stream);
    if (_distinct_keys.load(std::memory_order_relaxed) == 0) {
      thrust::fill(rmm::exec_policy_nosync(stream, temp_mr),
                   targets.begin(),
                   targets.end(),
                   cudf::detail::CUDF_SIZE_TYPE_SENTINEL);
    } else if (_has_nested_keys) {
      find_existing<true>(keys, row_bitmask, targets.data(), stream);
    } else {
      find_existing<false>(keys, row_bitmask, targets.data(), stream);
    }
    _insert_done.record(stream);

    rmm::device_buffer unmatched(static_cast<std::size_t>(size) * sizeof(size_type), stream, mr);
    auto const unmatched_data  = static_cast<size_type*>(unmatched.data());
    auto const end             = thrust::copy_if(rmm::exec_policy(stream, temp_mr),
                                     cuda::counting_iterator<size_type>{0},
                                     cuda::counting_iterator<size_type>{size},
                                     unmatched_data,
                                     is_unmatched_fn{targets.data(), row_bitmask});
    auto const unmatched_count = static_cast<size_type>(end - unmatched_data);

    if (_distinct_keys.load(std::memory_order_relaxed) != 0) {
      _results_read_done.wait(stream);
      compute_warp_aggregates(data.select(_value_col_indices),
                              _agg_kinds,
                              {targets.data(), targets.size()},
                              _agg_results->mutable_view(),
                              stream,
                              temp_mr);
      record_aggregate_completion(stream);
    }
    if (unmatched_count == 0) { return make_empty_column(data_type{type_id::INT32}); }
    return std::make_unique<column>(data_type{type_id::INT32},
                                    unmatched_count,
                                    std::move(unmatched),
                                    cudf::create_null_mask(0, mask_state::UNALLOCATED, stream, mr),
                                    0);
  } catch (...) {
    // Complete any enqueued reads or updates before later calls can resize state.
    static_cast<void>(cudaStreamSynchronize(stream.get()));
    throw;
  }
}

}  // namespace cudf::groupby
