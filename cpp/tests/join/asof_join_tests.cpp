/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>

#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/join/asof_join.hpp>
#include <cudf/join/join.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/device_uvector.hpp>

#include <algorithm>
#include <vector>

namespace {

using cudf::test::fixed_width_column_wrapper;
using cudf::test::strings_column_wrapper;

class AsofJoinTest : public cudf::test::BaseFixture {};

void expect_gather_map(rmm::device_uvector<cudf::size_type> const& actual,
                       std::vector<cudf::size_type> const& expected)
{
  auto const stream      = cudf::get_default_stream();
  auto const host_actual = cudf::detail::make_host_vector_async(actual, stream);
  stream.sync();

  ASSERT_EQ(host_actual.size(), expected.size());
  EXPECT_TRUE(std::equal(host_actual.begin(), host_actual.end(), expected.begin()));
}

TEST_F(AsofJoinTest, Ungrouped)
{
  fixed_width_column_wrapper<int32_t> right_on{1, 3, 3, 7};
  fixed_width_column_wrapper<int32_t> left_on{0, 1, 2, 3, 8};

  cudf::asof_join join(cudf::table_view{}, right_on);
  auto result = join.join(cudf::table_view{}, left_on, cudf::asof_join_strategy::BACKWARD, true);

  expect_gather_map(*result, {cudf::JoinNoMatch, 0, 0, 2, 3});
}

TEST_F(AsofJoinTest, SingleStringGroupingKey)
{
  strings_column_wrapper right_by{"a", "a", "b", "b"};
  fixed_width_column_wrapper<int32_t> right_on{1, 3, 2, 4};
  strings_column_wrapper left_by{"b", "a", "c", "a", "b"};
  fixed_width_column_wrapper<int32_t> left_on{3, 0, 5, 3, 5};

  cudf::asof_join join(cudf::table_view{{right_by}}, right_on);
  auto result =
    join.join(cudf::table_view{{left_by}}, left_on, cudf::asof_join_strategy::BACKWARD, true);

  expect_gather_map(*result, {2, cudf::JoinNoMatch, cudf::JoinNoMatch, 1, 3});
}

TEST_F(AsofJoinTest, MultipleGroupingKeys)
{
  strings_column_wrapper right_by_1{"a", "a", "a", "b", "b"};
  fixed_width_column_wrapper<int32_t> right_by_2{1, 1, 2, 1, 1};
  fixed_width_column_wrapper<int32_t> right_on{1, 4, 2, 3, 5};
  strings_column_wrapper left_by_1{"b", "a", "a"};
  fixed_width_column_wrapper<int32_t> left_by_2{1, 2, 1};
  fixed_width_column_wrapper<int32_t> left_on{4, 1, 4};

  cudf::asof_join join(cudf::table_view{{right_by_1, right_by_2}}, right_on);
  auto result = join.join(
    cudf::table_view{{left_by_1, left_by_2}}, left_on, cudf::asof_join_strategy::BACKWARD, true);

  expect_gather_map(*result, {3, cudf::JoinNoMatch, 1});
}

TEST_F(AsofJoinTest, NullOrderedKeys)
{
  fixed_width_column_wrapper<int32_t> right_on({0, 1, 3}, cudf::test::iterators::nulls_at({0}));
  fixed_width_column_wrapper<int32_t> left_on({0, 1, 2, 3}, cudf::test::iterators::nulls_at({3}));

  cudf::asof_join join(cudf::table_view{}, right_on);
  auto result = join.join(cudf::table_view{}, left_on, cudf::asof_join_strategy::BACKWARD, true);

  expect_gather_map(*result, {cudf::JoinNoMatch, 1, 1, cudf::JoinNoMatch});
}

TEST_F(AsofJoinTest, NullGroupingKeysDoNotMatch)
{
  strings_column_wrapper right_by({"", "", "a"}, cudf::test::iterators::nulls_at({0, 1}));
  fixed_width_column_wrapper<int32_t> right_on{1, 2, 1};
  strings_column_wrapper left_by({"", "a"}, cudf::test::iterators::nulls_at({0}));
  fixed_width_column_wrapper<int32_t> left_on{2, 1};

  cudf::asof_join join(cudf::table_view{{right_by}}, right_on);
  auto result =
    join.join(cudf::table_view{{left_by}}, left_on, cudf::asof_join_strategy::BACKWARD, true);

  expect_gather_map(*result, {cudf::JoinNoMatch, 2});
}

TEST_F(AsofJoinTest, EmptyInputs)
{
  fixed_width_column_wrapper<int32_t> empty{};
  fixed_width_column_wrapper<int32_t> left_on{1, 2};

  cudf::asof_join empty_right_join(cudf::table_view{}, empty);
  auto no_matches =
    empty_right_join.join(cudf::table_view{}, left_on, cudf::asof_join_strategy::BACKWARD, true);
  expect_gather_map(*no_matches, {cudf::JoinNoMatch, cudf::JoinNoMatch});

  cudf::asof_join join(cudf::table_view{}, left_on);
  auto empty_result =
    join.join(cudf::table_view{}, empty, cudf::asof_join_strategy::BACKWARD, true);
  expect_gather_map(*empty_result, {});
}

TEST_F(AsofJoinTest, TimestampOrderedKey)
{
  fixed_width_column_wrapper<cudf::timestamp_ms, int64_t> right_on{1, 3, 5};
  fixed_width_column_wrapper<cudf::timestamp_ms, int64_t> left_on{2, 5};

  cudf::asof_join join(cudf::table_view{}, right_on);
  auto result = join.join(cudf::table_view{}, left_on, cudf::asof_join_strategy::BACKWARD, true);

  expect_gather_map(*result, {0, 2});
}

TEST_F(AsofJoinTest, UnsupportedOptions)
{
  fixed_width_column_wrapper<int32_t> right_on{1};
  fixed_width_column_wrapper<int32_t> left_on{1};
  cudf::asof_join join(cudf::table_view{}, right_on);

  EXPECT_THROW(
    (void)join.join(cudf::table_view{}, left_on, cudf::asof_join_strategy::FORWARD, true),
    cudf::logic_error);
  EXPECT_THROW(
    (void)join.join(cudf::table_view{}, left_on, cudf::asof_join_strategy::BACKWARD, false),
    cudf::logic_error);
}

TEST_F(AsofJoinTest, MismatchedOrderedKeyTypes)
{
  fixed_width_column_wrapper<int32_t> right_on{1};
  fixed_width_column_wrapper<int64_t> left_on{1};
  cudf::asof_join join(cudf::table_view{}, right_on);

  EXPECT_THROW(
    (void)join.join(cudf::table_view{}, left_on, cudf::asof_join_strategy::BACKWARD, true),
    cudf::logic_error);
}

}  // namespace
