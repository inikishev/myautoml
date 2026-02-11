import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal, assert_series_equal

from myautoml.utils.polars_utils import (
    include_exclude_cols,
    maybe_stack,
    to_dataframe,
    to_lazyframe,
    with_columns_nonstrict,
)


class TestToDataframe:
    def test_to_dataframe_with_polars_dataframe(self):
        """Test conversion with polars DataFrame"""
        df = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
        result = to_dataframe(df)
        assert isinstance(result, pl.DataFrame)
        assert_frame_equal(result, df)

    def test_to_dataframe_with_polars_lazyframe(self):
        """Test conversion with polars LazyFrame"""
        lazy_df = pl.LazyFrame({"a": [1, 2], "b": [3, 4]})
        result = to_dataframe(lazy_df)
        assert isinstance(result, pl.DataFrame)
        assert_frame_equal(result, lazy_df.collect())

    def test_to_dataframe_with_polars_series(self):
        """Test conversion with polars Series"""
        series = pl.Series("col", [1, 2, 3])
        result = to_dataframe(series)
        expected = pl.DataFrame({"col": [1, 2, 3]})
        assert isinstance(result, pl.DataFrame)
        assert_frame_equal(result, expected)

    def test_to_dataframe_with_pandas_dataframe(self):
        """Test conversion with pandas DataFrame"""
        pd = pytest.importorskip("pandas")
        pd_df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        result = to_dataframe(pd_df)
        assert isinstance(result, pl.DataFrame)
        # Check that values match
        assert result["a"].to_list() == [1, 2]
        assert result["b"].to_list() == [3, 4]

    def test_to_dataframe_with_pandas_series(self):
        """Test conversion with pandas Series"""
        pd = pytest.importorskip("pandas")
        pd_series = pd.Series([1, 2, 3], name="col")
        result = to_dataframe(pd_series)
        assert isinstance(result, pl.DataFrame)
        assert result["col"].to_list() == [1, 2, 3]

    def test_to_dataframe_unsupported_type(self):
        """Test conversion with unsupported type raises TypeError"""
        with pytest.raises(TypeError, match="doesn't support objects of type"):
            to_dataframe([1, 2, 3])


class TestToLazyframe:
    def test_to_lazyframe_with_polars_lazyframe(self):
        """Test conversion with polars LazyFrame"""
        lazy_df = pl.LazyFrame({"a": [1, 2], "b": [3, 4]})
        result = to_lazyframe(lazy_df)
        assert isinstance(result, pl.LazyFrame)
        assert_frame_equal(result.collect(), lazy_df.collect())

    def test_to_lazyframe_with_polars_dataframe(self):
        """Test conversion with polars DataFrame"""
        df = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
        result = to_lazyframe(df)
        assert isinstance(result, pl.LazyFrame)
        assert_frame_equal(result.collect(), df)

    def test_to_lazyframe_with_polars_series(self):
        """Test conversion with polars Series"""
        series = pl.Series("col", [1, 2, 3])
        result = to_lazyframe(series)
        assert isinstance(result, pl.LazyFrame)
        expected = pl.DataFrame({"col": [1, 2, 3]}).lazy()
        assert_frame_equal(result.collect(), expected.collect())

    def test_to_lazyframe_with_pandas_dataframe(self):
        """Test conversion with pandas DataFrame"""
        pd = pytest.importorskip("pandas")
        pd_df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        result = to_lazyframe(pd_df)
        assert isinstance(result, pl.LazyFrame)
        collected_result = result.collect()
        assert collected_result["a"].to_list() == [1, 2]
        assert collected_result["b"].to_list() == [3, 4]

    def test_to_lazyframe_with_pandas_series(self):
        """Test conversion with pandas Series"""
        pd = pytest.importorskip("pandas")
        pd_series = pd.Series([1, 2, 3], name="col")
        result = to_lazyframe(pd_series)
        assert isinstance(result, pl.LazyFrame)
        collected_result = result.collect()
        assert collected_result["col"].to_list() == [1, 2, 3]

    def test_to_lazyframe_unsupported_type(self):
        """Test conversion with unsupported type raises TypeError"""
        with pytest.raises(TypeError, match="doesn't support objects of type"):
            to_lazyframe([1, 2, 3])


class TestMaybeStack:
    def test_maybe_stack_empty_args(self):
        """Test maybe_stack with no arguments"""
        result = maybe_stack()
        assert result is None

    def test_maybe_stack_all_none(self):
        """Test maybe_stack with all None arguments"""
        result = maybe_stack(None, None, None)
        assert result is None

    def test_maybe_stack_single_dataframe(self):
        """Test maybe_stack with a single DataFrame"""
        df = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
        result = maybe_stack(df)
        assert isinstance(result, pl.DataFrame)
        assert_frame_equal(result, df)

    def test_maybe_stack_multiple_dataframes(self):
        """Test maybe_stack with multiple DataFrames"""
        df1 = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
        df2 = pl.DataFrame({"a": [5, 6], "b": [7, 8]})
        result = maybe_stack(df1, df2)
        expected = pl.concat([df1, df2])
        assert isinstance(result, pl.DataFrame)
        assert_frame_equal(result, expected)

    def test_maybe_stack_with_none_values(self):
        """Test maybe_stack with some None values"""
        df1 = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
        df2 = pl.DataFrame({"a": [5, 6], "b": [7, 8]})
        result = maybe_stack(df1, None, df2)
        expected = pl.concat([df1, df2])
        assert isinstance(result, pl.DataFrame)
        assert_frame_equal(result, expected)

    def test_maybe_stack_lazyframe(self):
        """Test maybe_stack with LazyFrames"""
        lf1 = pl.LazyFrame({"a": [1, 2], "b": [3, 4]})
        lf2 = pl.LazyFrame({"a": [5, 6], "b": [7, 8]})
        result = maybe_stack(lf1, lf2)
        expected = pl.concat([lf1, lf2])
        assert isinstance(result, pl.LazyFrame)
        assert_frame_equal(result.collect(), expected.collect())


class TestIncludeExcludeCols:
    def test_include_exclude_cols_dataframe_include_only(self):
        """Test include_exclude_cols with DataFrame and include only"""
        df = pl.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
        result = include_exclude_cols(df, include=["a", "c"], exclude=None)
        expected = pl.DataFrame({"a": [1, 2], "c": [5, 6]})
        assert_frame_equal(result, expected)

    def test_include_exclude_cols_dataframe_exclude_only(self):
        """Test include_exclude_cols with DataFrame and exclude only"""
        df = pl.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
        result = include_exclude_cols(df, include=None, exclude=["b"])
        expected = pl.DataFrame({"a": [1, 2], "c": [5, 6]})
        assert_frame_equal(result, expected)

    def test_include_exclude_cols_dataframe_include_and_exclude(self):
        """Test include_exclude_cols with DataFrame and both include and exclude"""
        df = pl.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6], "d": [7, 8]})
        result = include_exclude_cols(df, include=["a", "b", "c"], exclude=["b"])
        expected = pl.DataFrame({"a": [1, 2], "c": [5, 6]})
        assert_frame_equal(result, expected)

    def test_include_exclude_cols_lazyframe_include_only(self):
        """Test include_exclude_cols with LazyFrame and include only"""
        lazy_df = pl.LazyFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
        result = include_exclude_cols(lazy_df, include=["a", "c"], exclude=None)
        expected = pl.LazyFrame({"a": [1, 2], "c": [5, 6]})
        assert_frame_equal(result.collect(), expected.collect())

    def test_include_exclude_cols_lazyframe_exclude_only(self):
        """Test include_exclude_cols with LazyFrame and exclude only"""
        lazy_df = pl.LazyFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
        result = include_exclude_cols(lazy_df, include=None, exclude=["b"])
        expected = pl.LazyFrame({"a": [1, 2], "c": [5, 6]})
        assert_frame_equal(result.collect(), expected.collect())

    def test_include_exclude_cols_lazyframe_include_and_exclude(self):
        """Test include_exclude_cols with LazyFrame and both include and exclude"""
        lazy_df = pl.LazyFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6], "d": [7, 8]})
        result = include_exclude_cols(lazy_df, include=["a", "b", "c"], exclude=["b"])
        expected = pl.LazyFrame({"a": [1, 2], "c": [5, 6]})
        assert_frame_equal(result.collect(), expected.collect())

    def test_include_exclude_cols_with_selector(self):
        """Test include_exclude_cols with column selectors"""
        df = pl.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
        # Test include with selector
        result = include_exclude_cols(df, include=pl.col("a"), exclude=None) # type:ignore
        expected = pl.DataFrame({"a": [1, 2]})
        assert_frame_equal(result, expected)

        # Test exclude with selector on original df
        result2 = include_exclude_cols(df, include=None, exclude=pl.col("b")) # type:ignore
        expected2 = pl.DataFrame({"a": [1, 2], "c": [5, 6]})
        assert_frame_equal(result2, expected2)


class TestWithColumnsNonstrict:
    def test_with_columns_nonstrict_basic(self):
        """Test with_columns_nonstrict with basic expressions"""
        lazy_df = pl.LazyFrame({"a": [1, 2], "b": [3, 4]})
        # Since 'c' is not in the original schema, it won't be added
        exprs = {"c": pl.col("a") + pl.col("b")}
        result = with_columns_nonstrict(lazy_df, exprs)
        # The result should be the same as the original since 'c' is not in schema
        expected = lazy_df
        assert_frame_equal(result.collect(), expected.collect())

        # Test with a column that exists in the schema (this would be unusual but for completeness)
        # Actually, the function filters expressions based on keys that exist in the schema
        # So let's test with a column name that matches an existing column name
        exprs2 = {"a": pl.col("a") + pl.lit(10)}  # 'a' exists in schema
        result2 = with_columns_nonstrict(lazy_df, exprs2)
        expected2 = lazy_df.with_columns(a=pl.col("a") + pl.lit(10))
        assert_frame_equal(result2.collect(), expected2.collect())

    def test_with_columns_nonstrict_multiple_expressions(self):
        """Test with_columns_nonstrict with multiple expressions"""
        lazy_df = pl.LazyFrame({"a": [1, 2], "b": [3, 4]})
        # Only expressions with keys that exist in the schema will be applied
        exprs = {
            "a": pl.col("a") + pl.lit(10),  # 'a' exists in schema
            "b": pl.col("b") * pl.lit(2),   # 'b' exists in schema
            "c": pl.col("a") + pl.col("b")  # 'c' does not exist in schema
        }
        result = with_columns_nonstrict(lazy_df, exprs)
        # Only 'a' and 'b' expressions should be applied
        expected = lazy_df.with_columns(
            a=pl.col("a") + pl.lit(10),
            b=pl.col("b") * pl.lit(2)
        )
        assert_frame_equal(result.collect(), expected.collect())

    def test_with_columns_nonstrict_with_sequence(self):
        """Test with_columns_nonstrict with sequence of expressions"""
        lazy_df = pl.LazyFrame({"a": [1, 2], "b": [3, 4]})
        # Since 'c' doesn't exist in schema, this should not add anything
        exprs = {
            "c": [pl.col("a") + pl.col("b"), pl.col("a") - pl.col("b")]  # 'c' doesn't exist in schema
        }
        result = with_columns_nonstrict(lazy_df, exprs)
        expected = lazy_df  # Should remain unchanged
        assert_frame_equal(result.collect(), expected.collect())

        # Test with existing column names that result in new columns (aliases)
        exprs2 = {
            "new_a": [pl.col("a") + pl.lit(5), pl.col("a") * pl.lit(2)]  # 'new_a' doesn't exist in schema
        }
        # This will still not be added because 'new_a' is not in the original schema
        result2 = with_columns_nonstrict(lazy_df, exprs2)
        assert_frame_equal(result2.collect(), lazy_df.collect())

        # Test with single expressions for existing columns (should work)
        exprs3 = {
            "a": [pl.col("a") + pl.lit(5)],  # Modify existing 'a' column
            "b": [pl.col("b") * pl.lit(3)]   # Modify existing 'b' column
        }
        result3 = with_columns_nonstrict(lazy_df, exprs3)
        expected3 = lazy_df.with_columns(
            (pl.col("a") + pl.lit(5)).alias("a"),  # This will modify 'a'
            (pl.col("b") * pl.lit(3)).alias("b")   # This will modify 'b'
        )
        result3_collected = result3.collect()
        expected3_collected = expected3.collect()
        assert_frame_equal(result3_collected, expected3_collected)

    def test_with_columns_nonstrict_nonexistent_column(self):
        """Test with_columns_nonstrict ignores non-existent columns"""
        lazy_df = pl.LazyFrame({"a": [1, 2], "b": [3, 4]})
        exprs = {
            "c": pl.col("a") + pl.col("b"),  # 'c' doesn't exist in schema
            "a": pl.col("a") * 2           # 'a' exists in schema
        }
        result = with_columns_nonstrict(lazy_df, exprs)
        # Only the 'a' column expression should be applied since 'a' exists in schema
        expected = lazy_df.with_columns(a=pl.col("a") * 2)
        assert_frame_equal(result.collect(), expected.collect())

