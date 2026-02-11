import pytest
import polars as pl
import polars.testing as pl_testing
from myautoml.polars_transformers.impute import MissingIndicator, MissingStatistics, SimpleImputer

@pytest.fixture
def sample_data():
    return pl.DataFrame({
        "a": [1, 2, None, 4],
        "b": [None, "y", "y", "z"],
        "c": [10.0, 20.0, 30.0, 40.0],  # No nulls
    })

def test_missing_indicator(sample_data):
    # Fit and transform
    mi = MissingIndicator(suffix="_nan")
    transformed = mi.fit_transform(sample_data).collect()

    # Check that indicator columns were added for 'a' and 'b', but not 'c'
    assert "a_nan" in transformed.columns
    assert "b_nan" in transformed.columns
    assert "c_nan" not in transformed.columns

    # Check values
    assert transformed["a_nan"].to_list() == [0, 0, 1, 0]
    assert transformed["b_nan"].to_list() == [1, 0, 0, 0]

    # Test inverse
    inverted = mi.inverse_transform(transformed).collect()
    pl_testing.assert_frame_equal(inverted, transformed.drop(["a_nan", "b_nan"]))

def test_missing_statistics(sample_data):
    ms = MissingStatistics(col_name="null_frac")
    transformed = ms.fit_transform(sample_data).collect()

    # Row 0: 1 null out of 3 columns (a, b, c) -> 0.333...
    # Row 2: 1 null out of 3 columns -> 0.333...
    # Row 3: 0 nulls -> 0.0
    expected_frac = [1/3, 0/3, 1/3, 0/3]
    assert transformed["null_frac"].to_list() == pytest.approx(expected_frac)

    # Test include/exclude
    ms_limited = MissingStatistics(include=["a", "b"], col_name="ab_null_frac")
    transformed_lim = ms_limited.fit_transform(sample_data).collect()
    # Row 0: a=1, b=None -> 0.5
    assert transformed_lim["ab_null_frac"][0] == 0.5

def test_simple_imputer_numeric_strategies(sample_data):
    # Test Median (numeric) + Mode (categorical fallback)
    si = SimpleImputer(strategy="median")
    transformed = si.fit_transform(sample_data).collect()

    # Column 'a': median of [1, 2, 4] is 2
    assert transformed["a"][2] == 2
    # Column 'b': mode of ["y", "y", "z"] is "y"
    assert transformed["b"][0] == "y"

    # Test Mean
    si_mean = SimpleImputer(strategy="mean")
    transformed_mean = si_mean.fit_transform(sample_data).collect()
    # (1 + 2 + 4) / 3 = 2.333...
    assert transformed_mean["a"][2] == pytest.approx(2.3333333)

def test_simple_imputer_constant():
    df = pl.DataFrame({"a": [1, None], "b": ["x", None]})
    si = SimpleImputer(strategy="constant", fill_value=99)
    transformed = si.fit_transform(df).collect()

    assert transformed["a"].to_list() == [1, 99]
    # Note: Because fill_value is 99 (int), and 'b' is string,
    # Polars might cast or fail depending on strictly.
    # In SimpleImputer, it uses pl.col(k).fill_null(v).
    assert transformed["b"][1] == "99" # Polars auto-casts to string for the fill

def test_simple_imputer_with_indicator(sample_data):
    si = SimpleImputer(strategy="mean", add_indicator=True)
    transformed = si.fit_transform(sample_data).collect()

    # Check both imputation and indicator existence
    assert "a__is_missing" in transformed.columns
    assert transformed["a"][2] != None
    assert transformed["a__is_missing"][2] == 1

    # Check inverse (should remove indicator but keep imputed values)
    inverted = si.inverse_transform(transformed).collect()
    assert "a__is_missing" not in inverted.columns
    assert inverted["a"][2] is not None

def test_simple_imputer_non_strict_behavior():
    df_train = pl.DataFrame({"a": [1, None], "b": [1, 2]})
    df_test = pl.DataFrame({"a": [None]}) # 'b' is missing

    si = SimpleImputer(strategy="constant", fill_value=0)
    si.fit(df_train)

    # Should not crash even though 'b' is missing from df_test
    transformed = si.transform(df_test).collect()
    assert transformed["a"][0] == 0
    assert "b" not in transformed.columns

def test_simple_imputer_mode_strategy():
    df = pl.DataFrame({
        "a": [1, 1, 2, None],
        "b": ["cat", "cat", "dog", None]
    })
    si = SimpleImputer(strategy="mode")
    transformed = si.fit_transform(df).collect()

    assert transformed["a"].to_list() == [1, 1, 2, 1]
    assert transformed["b"].to_list() == ["cat", "cat", "dog", "cat"]


def test_all_missing():
    df = pl.DataFrame({"a": [None, None, None]})

    # SimpleImputer won't impute unless strategy is "constant"
    si = SimpleImputer()
    transformed = si.fit_transform(df).collect()
    pl_testing.assert_frame_equal(df, transformed)

    si = SimpleImputer(strategy="constant")
    transformed = si.fit_transform(df).collect()
    assert transformed["a"].to_list() == [0, 0, 0]