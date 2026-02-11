from typing import Any

import polars as pl
import polars.testing as pl_testing
import pytest

from myautoml.polars_transformers.one_hot import OneHotEncoder


@pytest.fixture
def sample_data():
    return pl.DataFrame({
        "cat1": ["A", "B", "A", "C", None], # High cardinality (3+null)
        "cat2": ["M", "F", "M", "F", "M"],    # Binary
        "cat3": ["X", "X", "X", "X", "Y"],    # One very frequent, one rare
        "num": [1, 2, 3, 4, 5]                 # Numeric
    }).lazy()

def test_one_hot_basic(sample_data):
    # skip_binary=False to ensure cat2 is processed
    # propagate_nulls=False to treat None as a category
    encoder = OneHotEncoder(
        include=["cat1", "cat2"],
        drop_first='none',
        propagate_nulls=False
    )

    transformed = encoder.fit_transform(sample_data).collect()

    # Check that original columns are dropped
    assert "cat1" not in transformed.columns
    assert "cat2" not in transformed.columns

    # Check new columns exist (cat1 has A, B, C, None)
    assert "cat1__A" in transformed.columns
    assert "cat1__None" in transformed.columns
    assert "cat2__M" in transformed.columns

    # Check value logic for cat1__A
    # Row 0 and 2 are "A"
    assert transformed["cat1__A"].to_list() == [1, 0, 1, 0, 0]
    # Row 4 is None
    assert transformed["cat1__None"].to_list() == [0, 0, 0, 0, 1]

def test_one_hot_propagate_nulls(sample_data):
    encoder = OneHotEncoder(include=["cat1"], propagate_nulls=True)
    transformed = encoder.fit_transform(sample_data).collect()

    # When propagate_nulls is True, the row where cat1 was None should have Null in all OH columns
    # cat1 row 4 was None.
    assert transformed.row(4, named=True)["cat1__A"] is None
    assert transformed.row(4, named=True)["cat1__B"] is None


def test_one_hot_min_frequency(sample_data):
    # cat3 has "X" (4 times) and "Y" (1 time). Total 5.
    # If min_frequency = 2, "Y" should be infrequent.
    encoder = OneHotEncoder(
        include=["cat3"],
        min_frequency=2,
        drop_first='none',
    )
    transformed = encoder.fit_transform(sample_data).collect()
    assert "cat3__X" in transformed.columns
    assert "cat3____infrequent" in transformed.columns
    assert "cat3__Y" not in transformed.columns

    # Row 4 was "Y", so it should be in infrequent
    assert transformed["cat3____infrequent"].to_list() == [0, 0, 0, 0, 1]

def test_one_hot_max_categories(sample_data):
    # cat1 has A(2), B(1), C(1). Max categories 1 should only keep "A"
    encoder = OneHotEncoder(include=["cat1"], max_categories=1, drop_first='none',)
    transformed = encoder.fit_transform(sample_data).collect()

    assert "cat1__A" in transformed.columns
    assert "cat1__B" not in transformed.columns
    assert "cat1____infrequent" in transformed.columns

def test_inverse_transform(sample_data):
    encoder = OneHotEncoder(include=["cat1"], propagate_nulls=False)

    transformed = encoder.fit_transform(sample_data)
    inverted = encoder.inverse_transform(transformed).collect()

    # Check that inverted matches original (excluding columns not processed)
    # Note: select to ensure column order is same for comparison
    original = sample_data.collect()
    pl_testing.assert_frame_equal(inverted, original)

def test_inverse_transform_with_infrequent(sample_data):
    # Note: Inverse transform maps infrequent to None as per docstring
    encoder = OneHotEncoder(include=["cat3"], min_frequency=2, drop_first='none',)

    transformed = encoder.fit_transform(sample_data)
    inverted = encoder.inverse_transform(transformed).collect()
    # The 'Y' in cat3 (row 4) should now be None
    expected_cat3 = ["X", "X", "X", "X", "__infrequent"]
    assert inverted["cat3"].to_list() == expected_cat3

def test_one_hot_empty_selectors(sample_data):
    # Test that it doesn't crash if no columns are selected
    encoder = OneHotEncoder(include=[])
    transformed = encoder.fit_transform(sample_data).collect()
    pl_testing.assert_frame_equal(transformed, sample_data.collect())

def test_one_hot_exclude(sample_data):
    # Exclude everything
    encoder = OneHotEncoder(include=None, exclude=["cat1", "cat2", "cat3", "num"])
    transformed = encoder.fit_transform(sample_data).collect()
    pl_testing.assert_frame_equal(transformed, sample_data.collect())

def test_unseen_categories():
    """Categories present in transform but not in fit should result in all zeros."""
    train = pl.DataFrame({"cat": ["A", "B"]})
    test = pl.DataFrame({"cat": ["A", "C", None]})

    encoder = OneHotEncoder(include=["cat"], drop_first='none', propagate_nulls=False)
    encoder.fit(train)
    transformed = encoder.transform(test).collect()

    # "C" was not in train, so cat__A and cat__B should be 0
    assert transformed.row(0, named=True) == {"cat__A": 1, "cat__B": 0}
    # Row 1 is "C"
    assert transformed.row(1, named=True) == {"cat__A": 0, "cat__B": 0}
    # Row 2 is None (also unseen in fit)
    assert transformed.row(2, named=True) == {"cat__A": 0, "cat__B": 0}

def test_single_unique_value():
    """Test that it skips columns with single unique value."""
    df = pl.DataFrame({"cat": ["A", "A", "A"]}).lazy()
    # skip_binary=True should skip columns with 2 OR LESS unique values.
    encoder = OneHotEncoder(include=["cat"])
    transformed = encoder.fit_transform(df).collect()

    pl_testing.assert_frame_equal(transformed, df.collect())

def test_single_null_value():
    """Test behavior when a column contains only nulls."""
    df = pl.DataFrame({"cat": [None, None]}).lazy()
    encoder = OneHotEncoder(include=["cat"], drop_first='none')

    # Should not crash. One category: "None"
    transformed = encoder.fit_transform(df).collect()

    pl_testing.assert_frame_equal(transformed, df.collect())

# --- 2. FREQUENCY & CATEGORY LIMITS ---

def test_min_frequency_float_threshold():
    """Test relative frequency (float) calculation."""
    # 10 rows total. 0.3 frequency = 3 occurrences required.
    df = pl.DataFrame({
        "cat": ["A", "A", "A", "B", "B", "C", "D", "E", "F", "G"]
    }).lazy()

    encoder = OneHotEncoder(
        include=["cat"],
        min_frequency=0.3,
        drop_first='none',
    )
    transformed = encoder.fit_transform(df).collect()

    # Only "A" has 3 instances. Everything else is infrequent.
    assert "cat__A" in transformed.columns
    assert "cat__B" not in transformed.columns
    assert "cat____infrequent" in transformed.columns
    # Row 3 (value "B") should be in infrequent
    assert transformed["cat____infrequent"][3] == 1

def test_max_categories_with_nulls():
    """Verify max_categories interaction with nulls when propagate_nulls is False."""
    df = pl.DataFrame({"cat": ["A", "A", "B", "C", None, None, None]}).lazy()
    # Frequencies: None(3), A(2), B(1), C(1)
    # If max_categories=2, we keep None and A.
    encoder = OneHotEncoder(
        include=["cat"],
        max_categories=2,
        propagate_nulls=False,
    )
    transformed = encoder.fit_transform(df).collect()

    assert "cat__None" in transformed.columns
    assert "cat__A" in transformed.columns
    assert "cat__B" not in transformed.columns
    assert "cat____infrequent" in transformed.columns

# --- 3. DATA TYPES ---

def test_numeric_one_hot():
    """One-hot encoding should work on numeric columns if explicitly included."""
    df = pl.DataFrame({"num_cat": [10, 20, 10, 30]}).lazy()
    encoder = OneHotEncoder(include=["num_cat"], drop_first='none',)
    transformed = encoder.fit_transform(df).collect()

    assert "num_cat__10" in transformed.columns
    assert transformed["num_cat__10"].to_list() == [1, 0, 1, 0]

def test_categorical_dtype_preservation():
    """Inverse transform should preserve the original Polars dtype (e.g. Categorical)."""
    df = pl.DataFrame({"cat": ["p", "q", "p"]}, schema={"cat": pl.Categorical}).lazy()
    encoder = OneHotEncoder(include=["cat"], drop_first='none',)

    transformed = encoder.fit_transform(df)
    inverted = encoder.inverse_transform(transformed).collect()

    assert inverted["cat"].dtype == pl.Categorical
    pl_testing.assert_frame_equal(inverted, df.collect())

# --- 4. INVERSE TRANSFORM LOGIC ---

def test_inverse_transform_propagate_nulls_true():
    """When propagate_nulls=True, ensure inverse transform restores None correctly."""
    df = pl.DataFrame({"cat": ["A", None, "B"]}).lazy()
    encoder = OneHotEncoder(include=["cat"], propagate_nulls=True, drop_first='none',)

    transformed = encoder.fit_transform(df)
    # In transform, the middle row will have all nulls.
    # The inverse transform must handle the case where all OH columns are null.
    inverted = encoder.inverse_transform(transformed).collect()

    assert inverted["cat"].to_list() == ["A", None, "B"]

def test_inverse_transform_overlap_prevention():
    """Ensure inverse transform doesn't accidentally drop user columns with similar names."""
    df = pl.DataFrame({
        "cat": ["A", "B"],
        "cat__A_extra": [99, 88] # Column that looks like an OH column
    }).lazy()

    encoder = OneHotEncoder(include=["cat"], drop_first='none',)
    transformed = encoder.fit_transform(df)
    inverted = encoder.inverse_transform(transformed).collect()

    assert "cat__A_extra" in inverted.columns
    assert inverted["cat__A_extra"].to_list() == [99, 88]

# --- 5. SYSTEM & UTILS ---

def test_maintain_order_determinism():
    """Check that column order is deterministic if maintain_order is True."""
    df = pl.DataFrame({"cat": ["C", "A", "B", "D"]}).lazy()

    encoder1 = OneHotEncoder(include=["cat"], maintain_order=True, drop_first='none',)
    cols1 = encoder1.fit_transform(df).collect().columns

    encoder2 = OneHotEncoder(include=["cat"], maintain_order=True, drop_first='none',)
    cols2 = encoder2.fit_transform(df).collect().columns

    assert cols1 == cols2

def test_input_types_conversion():
    """Test that the encoder handles pandas DataFrames if passed (via utils)."""
    try:
        import pandas as pd
        pdf = pd.DataFrame({"cat": ["A", "B", "A"]})
        encoder = OneHotEncoder(include=["cat"], drop_first='none',)
        # Should convert pd -> pl.LazyFrame internally
        transformed = encoder.fit_transform(pdf).collect()
        assert "cat__A" in transformed.columns
    except ImportError:
        pytest.skip("Pandas not installed")

def test_fit_empty_dataframe():
    """Fit should not crash on a dataframe with no rows but a schema."""
    df = pl.DataFrame(schema={"cat": pl.String}).lazy()
    encoder = OneHotEncoder(include=["cat"])
    encoder.fit(df)
    transformed = encoder.transform(df) # No data to find categories
    pl_testing.assert_frame_equal(df.collect(), transformed.collect())
    inverse = encoder.inverse_transform(transformed)
    pl_testing.assert_frame_equal(df.collect(), inverse.collect())

def test_include_none_logic():
    """If include is None, it should process all columns or respect exclude."""
    df = pl.DataFrame({
        "cat1": ["A", "B", "A"],
        "cat2": ["X", "Y", "X"]
    }).lazy()

    encoder = OneHotEncoder(include=None, exclude=["cat2"], drop_first='none',)
    transformed = encoder.fit_transform(df).collect()

    assert "cat1__A" in transformed.columns
    assert "cat2" in transformed.columns # Was excluded from being OH'd, so remains as is
    assert "cat2__X" not in transformed.columns



@pytest.fixture
def sample_df():
    return pl.DataFrame({
        "color": ["red", "blue", "green", "red", "blue", "red"], # 3 cats
        "size": ["S", "S", "M", "L", "M", "S"],                # 3 cats
        "binary": ["yes", "no", "yes", "yes", "no", "no"],     # 2 cats
        "const": ["A", "A", "A", "A", "A", "A"],               # 1 cat (should skip)
        "with_null": ["cold", "hot", None, "cold", "hot", None]
    })

class TestOneHotEncoder:

    def test_basic_encoding(self, sample_df):
        # drop_first="none" is easiest to verify first
        encoder = OneHotEncoder(include=["color"], drop_first="none")
        encoder.fit(sample_df)
        out = encoder.transform(sample_df).collect()

        # Check if original column is gone and new columns exist
        assert "color" not in out.columns
        assert all(c in out.columns for c in ["color__red", "color__blue", "color__green"])
        # Check sum of rows is 1 (standard OHE)
        assert all(out.select(pl.sum_horizontal(pl.col("^color_.*$")) == 1).to_series())

    def test_drop_first_binary(self, sample_df):
        # 'binary' has 2 categories, 'color' has 3.
        # drop_first="binary" should drop 1 column for 'binary', but 0 for 'color'.
        encoder = OneHotEncoder(include=["color", "binary"], drop_first="binary")
        encoder.fit(sample_df)
        out = encoder.transform(sample_df).collect()

        # Color should have 3 cols (not binary)
        color_cols = [c for c in out.columns if c.startswith("color_")]
        assert len(color_cols) == 3

        # Binary should have 1 col (is binary)
        binary_cols = [c for c in out.columns if c.startswith("binary_")]
        assert len(binary_cols) == 1

    def test_drop_first_all(self, sample_df):
        encoder = OneHotEncoder(include=["color"], drop_first="all")
        encoder.fit(sample_df)
        out = encoder.transform(sample_df).collect()

        # Color has 3 unique values, should result in 2 columns
        color_cols = [c for c in out.columns if c.startswith("color_")]
        assert len(color_cols) == 2

    def test_constant_column_skipping(self, sample_df):
        encoder = OneHotEncoder(include=["const"])
        encoder.fit(sample_df)
        out = encoder.transform(sample_df).collect()

        # Docstring says it skips constant columns.
        # Check if 'const' remains or is removed without OHE.
        assert "const" in out.columns
        assert not any(c.startswith("const_") for c in out.columns)

    def test_propagate_nulls_true(self, sample_df):
        encoder = OneHotEncoder(include=["with_null"], propagate_nulls=True, drop_first="none")
        encoder.fit(sample_df)
        out = encoder.transform(sample_df).collect()

        # Rows where with_null was None should have null in all OHE columns
        null_indices = sample_df.get_column("with_null").is_null()
        ohe_cols = [c for c in out.columns if c.startswith("with_null_")]

        for col in ohe_cols:
            assert out.filter(null_indices).get_column(col).null_count() == 2

    def test_propagate_nulls_false(self, sample_df):
        encoder = OneHotEncoder(include=["with_null"], propagate_nulls=False, drop_first="none")
        encoder.fit(sample_df)
        out = encoder.transform(sample_df).collect()

        # Null should be treated as its own category (e.g., "with_null_null")
        assert any("null" in c.lower() for c in out.columns)
        # Sum of row values should be 1 even for rows that were null
        assert all(out.select(pl.sum_horizontal(pl.col("^with_null_.*$")) == 1).to_series())

    def test_max_categories(self, sample_df):
        # Color has red(3), blue(2), green(1)
        encoder = OneHotEncoder(include=["color"], max_categories=2, infrequent_value="other", drop_first="none")
        encoder.fit(sample_df)
        out = encoder.transform(sample_df).collect()

        # Should have red, blue, and 'other' (for green)
        assert "color__red" in out.columns
        assert "color__blue" in out.columns
        assert "color__other" in out.columns
        assert "color__green" not in out.columns

    def test_inverse_transform_roundtrip(self, sample_df):
        # Note: drop_first must be "none" for simple argmax inverse as per docstring
        cols = ["color", "size"]
        encoder = OneHotEncoder(include=cols, drop_first="none")

        transformed = encoder.fit(sample_df).transform(sample_df)
        inverted = encoder.inverse_transform(transformed).collect()

        # Check original columns are restored correctly
        for col in cols:
            pl_testing.assert_series_equal(inverted[col], sample_df[col])

    def test_unseen_categories(self, sample_df):
        encoder = OneHotEncoder(include=["color"], drop_first="none")
        encoder.fit(sample_df)

        new_df = pl.DataFrame({"color": ["purple"]}) # Purple never seen
        out = encoder.transform(new_df).collect()

        # Standard behavior: all zeros for unseen categories if no infrequent bundling
        ohe_cols = [c for c in out.columns if c.startswith("color_")]
        for col in ohe_cols:
            assert (out[col] == 0).all()