/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "common.cuh"
#include "groupby/common/utils.hpp"

#include <cudf/detail/gather.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/buffer>
#include <cuda/iterator>
#include <cuda/stream>
#include <thrust/for_each.h>

#include <string>

namespace cudf::groupby {

// A shared owner prevents flat and nested insertion from emitting the same hash-cache kernel.
void compute_batch_hashes(
  std::shared_ptr<cudf::detail::row::hash::preprocessed_table> const& preprocessed_batch,
  cudf::nullate::DYNAMIC has_null,
  bitmask_type const* batch_bitmask,
  hash_value_type* batch_hash_cache,
  size_type batch_size,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr);

template <bool has_nested>
streaming_groupby::impl::batch_insert_result streaming_groupby::impl::probe_and_insert_impl(
  table_view const& batch_keys, cuda::stream_ref stream)
{
  auto const batch_size = batch_keys.num_rows();
  auto const temp_mr    = cudf::get_current_device_resource_ref();
  auto const has_null   = cudf::nullate::DYNAMIC{_has_nullable_keys};

  // Preprocess batch for row operators.
  auto preprocessed_batch =
    cudf::detail::row::hash::preprocessed_table::create(batch_keys, stream, temp_mr);

  // Compute the null-exclusion bitmask first so the hash cache pass can skip
  // hashing rows that will be excluded
  auto const skip_rows_with_nulls = _has_nullable_keys && _null_handling == null_policy::EXCLUDE;
  auto [bitmask_buffer, batch_bitmask] =
    skip_rows_with_nulls
      ? detail::compute_row_bitmask(batch_keys, stream)
      : std::pair<cuda::device_buffer<std::byte>, bitmask_type const*>{
          cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED, stream), nullptr};

  // Precompute batch hash values.  Caching is faster than inlining the row hasher
  cuda::device_buffer<hash_value_type> batch_hash_cache(stream, temp_mr, batch_size, cuda::no_init);
  compute_batch_hashes(preprocessed_batch,
                       has_null,
                       batch_bitmask,
                       batch_hash_cache.data(),
                       batch_size,
                       stream,
                       temp_mr);

  // Pass 1 — fused insert_and_find + compact via `thrust::copy_if`.
  // Per-row work in the predicate: writes target_indices and slot_offsets.
  // Output: batch row indices of newly inserted keys, compacted into batch_local_indices.
  // Count: iterator distance returned by copy_if inside the helper.
  // slot_offsets stores 4-byte slot offsets (vs. 8-byte raw pointers) to halve temp memory.
  cuda::device_buffer<size_type> target_indices(stream, temp_mr, batch_size, cuda::no_init);
  cuda::device_buffer<size_type> slot_offsets(stream, temp_mr, batch_size, cuda::no_init);
  cuda::device_buffer<size_type> batch_local_indices(stream, temp_mr, batch_size, cuda::no_init);

  // First batch has no compacted batches yet, so all slot values are transient (>=
  // _max_distinct_keys) and only the batch-self equality branch of n_table_comparator
  // can fire.  Dispatch to a dedicated helper that uses first_batch_comparator to skip
  // the cross-table dispatch, the cross-comparator build, and the dense-ID branches.
  // The two helpers live in separate TUs to parallelize the heavy cuco/thrust template
  // instantiations.
  size_type const new_distinct_keys =
    _compacted_batches.empty()
      ? probe_and_insert_first_batch<has_nested>(preprocessed_batch,
                                                 has_null,
                                                 batch_bitmask,
                                                 batch_hash_cache.data(),
                                                 batch_size,
                                                 target_indices.data(),
                                                 slot_offsets.data(),
                                                 batch_local_indices.data(),
                                                 stream)
      : probe_and_insert_subsequent<has_nested>(preprocessed_batch,
                                                has_null,
                                                batch_bitmask,
                                                batch_hash_cache.data(),
                                                batch_size,
                                                target_indices.data(),
                                                slot_offsets.data(),
                                                batch_local_indices.data(),
                                                stream);
  auto const new_key_indices = cudf::device_span<size_type const>{
    batch_local_indices.data(), static_cast<std::size_t>(new_distinct_keys)};

  if (new_distinct_keys > 0) {
    // Bound check: the hash set has already been written above (transient slot values),
    // so on failure the object is left invalidated; further aggregate()/merge() calls
    // will throw immediately while finalize() can still recover partial results.
    auto const distinct_so_far = _distinct_keys.load(std::memory_order_relaxed);
    if (distinct_so_far + new_distinct_keys > _max_distinct_keys) {
      _invalidated = true;
      CUDF_FAIL("Distinct key count (" + std::to_string(distinct_so_far + new_distinct_keys) +
                ") would exceed max_distinct_keys (" + std::to_string(_max_distinct_keys) + ").");
    }

    // Gather compacted distinct keys from the batch.
    auto compacted = cudf::detail::gather(batch_keys,
                                          new_key_indices,
                                          out_of_bounds_policy::DONT_CHECK,
                                          cudf::negative_index_policy::NOT_ALLOWED,
                                          stream,
                                          temp_mr);

    auto preprocessed_compacted =
      cudf::detail::row::hash::preprocessed_table::create(compacted->view(), stream, temp_mr);

    // Store the compacted batch.
    auto const new_batch_id    = static_cast<size_type>(_compacted_batches.size());
    auto const dense_id_offset = distinct_so_far;
    _compacted_batches.push_back(std::move(compacted));
    _preprocessed_batches.push_back(preprocessed_compacted);

    // Pass 2 — fused: rewrites slots from transient encoding to dense IDs and
    // writes the {batch_id, row} companion entry.
    auto* const base = _key_set->data();
    thrust::for_each_n(rmm::exec_policy_nosync(stream, temp_mr),
                       cuda::counting_iterator<size_type>(0),
                       new_distinct_keys,
                       finalize_new_key_fn{batch_local_indices.data(),
                                           base,
                                           slot_offsets.data(),
                                           _key_loc->data(),
                                           new_batch_id,
                                           dense_id_offset});

    thrust::for_each_n(rmm::exec_policy_nosync(stream, temp_mr),
                       cuda::counting_iterator<size_type>(0),
                       batch_size,
                       update_transient_target_indices_fn{
                         base, slot_offsets.data(), _max_distinct_keys, target_indices.data()});

    _distinct_keys.fetch_add(new_distinct_keys, std::memory_order_relaxed);
  }
  // If new_distinct_keys == 0, target_indices is already final from Pass 1 — every
  // slot held a dense ID at probe time, so *iter was already the correct dense ID.

  return batch_insert_result{
    std::move(target_indices), new_distinct_keys, std::move(bitmask_buffer)};
}

template <typename SetRef>
struct find_existing_fn {
  SetRef set_ref;
  bitmask_type const* row_bitmask;
  size_type offset;
  size_type* target_indices;

  __device__ void operator()(size_type row) const
  {
    auto target = cudf::detail::CUDF_SIZE_TYPE_SENTINEL;
    if (!row_bitmask || cudf::bit_is_set(row_bitmask, row)) {
      auto const found = set_ref.find(offset + row);
      if (found != set_ref.end()) { target = *found; }
    }
    target_indices[row] = target;
  }
};

template <bool has_nested>
void streaming_groupby::impl::find_existing(table_view const& batch_keys,
                                            bitmask_type const* batch_bitmask,
                                            size_type* target_indices,
                                            cuda::stream_ref stream)
{
  auto const mr       = cudf::get_current_device_resource_ref();
  auto const has_null = cudf::nullate::DYNAMIC{_has_nullable_keys};
  auto preprocessed   = cudf::detail::row::hash::preprocessed_table::create(batch_keys, stream, mr);
  cuda::device_buffer<hash_value_type> hashes(stream, mr, batch_keys.num_rows(), cuda::no_init);
  compute_batch_hashes(
    preprocessed, has_null, batch_bitmask, hashes.data(), batch_keys.num_rows(), stream, mr);
  auto const self = cudf::detail::row::equality::self_comparator{preprocessed}.equal_to<has_nested>(
    has_null, null_equality::EQUAL);
  auto cross =
    build_cross_comparators<has_nested>(preprocessed, _preprocessed_batches, has_null, stream);
  auto const comparator =
    n_table_comparator{self, cross.data(), _key_loc->data(), _max_distinct_keys};
  auto const ref = _key_set->ref(cuco::op::find)
                     .rebind_hash_function(offset_cache_hasher{hashes.data(), _max_distinct_keys})
                     .rebind_key_eq(comparator);
  thrust::for_each_n(rmm::exec_policy_nosync(stream, mr),
                     cuda::counting_iterator<size_type>{0},
                     batch_keys.num_rows(),
                     find_existing_fn{ref, batch_bitmask, _max_distinct_keys, target_indices});
}

}  // namespace cudf::groupby
