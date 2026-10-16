/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "insert.cuh"

namespace cudf::groupby {

template cuda::device_buffer<size_type> streaming_groupby::impl::probe_and_insert_impl<true>(
  table_view const& batch_keys, bool use_partials, cuda::stream_ref stream);

template void streaming_groupby::impl::find_existing<true>(table_view const&,
                                                           bitmask_type const*,
                                                           size_type*,
                                                           cuda::stream_ref);

}  // namespace cudf::groupby
