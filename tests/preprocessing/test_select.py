import polars as pl
import pytest
from polars.testing import assert_frame_equal

from myautoml.polars_transformers.select import (
    DropCols,
    DropConstant,
    RemoveDuplicates,
    SelectCols,
)


@pytest.fixture
def sample_df():
    return pl.DataFrame({
        "a": [1, 2, 3],
        "b": [4, 5, 6],
        "c": [7, 7, 7],  # Constant
        "d": [1, 1, 1],  # Constant
    })

@pytest.fixture
def duplicate_df():
    return pl.DataFrame({
        "a": [1, 1, 2],
        "b": [1, 1, 3]
    })

class TestSelectCols:
    def test_select_cols_transform(self, sample_df):
        transformer = SelectCols("a", "b")
        # Test with DataFrame input
        result = transformer.fit_transform(sample_df)

        assert isinstance(result, pl.LazyFrame)
        assert result.collect_schema().names() == ["a", "b"]
        assert_frame_equal(result.collect(), sample_df.select("a", "b"))

    def test_select_cols_lazy(self, sample_df):
        transformer = SelectCols("c")
        result = transformer.transform(sample_df.lazy())
        assert isinstance(result, pl.LazyFrame)
        assert result.collect_schema().names() == ["c"]

class TestDropCols:
    def test_drop_cols_transform(self, sample_df):
        transformer = DropCols("a", "d")
        result = transformer.fit_transform(sample_df)

        assert isinstance(result, pl.LazyFrame)
        assert result.collect_schema().names() == ["b", "c"]
        assert_frame_equal(result.collect(), sample_df.drop("a", "d"))

class TestDropConstant:
    def test_drop_constant_fit_transform(self, sample_df):
        transformer = DropConstant()
        # 'c' and 'd' are constant
        result = transformer.fit_transform(sample_df)

        assert isinstance(result, pl.LazyFrame)
        remaining_cols = result.collect_schema().names()
        assert "a" in remaining_cols
        assert "b" in remaining_cols
        assert "c" not in remaining_cols
        assert "d" not in remaining_cols
        assert len(transformer.drop_cols_) == 2

    def test_drop_constant_inverse(self, sample_df):
        transformer = DropConstant()
        transformed = transformer.fit_transform(sample_df)
        inverted = transformer.inverse_transform(transformed)

        assert isinstance(inverted, pl.LazyFrame)
        # Check if original values and column order are restored
        assert_frame_equal(inverted.collect(), sample_df)

    def test_drop_constant_include_exclude(self, sample_df):
        # Only look at 'c', leave 'd' even though it is constant
        transformer = DropConstant(include=["a", "b", "c"])
        result = transformer.fit_transform(sample_df).collect()

        assert "c" not in result.columns
        assert "d" in result.columns

        # Exclude 'c' from being dropped
        transformer = DropConstant(exclude=["c"])
        result = transformer.fit_transform(sample_df).collect()
        assert "c" in result.columns
        assert "d" not in result.columns

class TestRemoveDuplicates:
    def test_remove_duplicates_basic(self, duplicate_df):
        transformer = RemoveDuplicates(maintain_order=True)
        result = transformer.fit_transform(duplicate_df).collect()

        assert result.height == 2
        assert_frame_equal(result, duplicate_df.unique(maintain_order=True))

    def test_remove_duplicates_subset(self):
        df = pl.DataFrame({
            "a": [1, 1, 2],
            "b": [1, 2, 3]
        })
        # If we only look at subset "a", row 2 is a duplicate of row 1
        transformer = RemoveDuplicates(subset=["a"], keep="first", maintain_order=True)
        result = transformer.fit_transform(df).collect()

        assert result.height == 2
        assert result["b"].to_list() == [1, 3]

def test_base_transform_interface():
    """Verify that the abstract base class logic works as expected."""
    class MockTransform(SelectCols):
        pass

    m = MockTransform("a")
    df = pl.DataFrame({"a": [1], "b": [2]})
    # Test fit returns self
    assert m.fit(df) is m
    # Test fit_transform logic
    assert m.fit_transform(df).collect().width == 1

def test_drop_constant_no_constants():
    df = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
    transformer = DropConstant()
    result = transformer.fit_transform(df).collect()
    assert result.width == 2
    assert transformer.drop_cols_ == []