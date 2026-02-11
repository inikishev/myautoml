import pandas as pd
import polars as pl
import pytest
from polars.testing import assert_frame_equal, assert_frame_not_equal

from myautoml.polars_transformers.ordinal import MapEncoder, OrdinalEncoder


@pytest.fixture
def sample_df():
    return pl.DataFrame({
        "cat": ["a", "b", "a", "c"],
        "num": [10, 20, 10, 30],
        "target": [0, 1, 0, 1]
    })

def test_ordinal_encoder_basic(sample_df):
    encoder = OrdinalEncoder(include=["cat", "num"])
    # Fit and transform
    result = encoder.fit_transform(sample_df).collect()

    # Check that columns are now integers
    assert result["cat"].dtype == pl.Int64
    assert result["num"].dtype == pl.Int64
    # Check that 'target' was untouched
    assert result["target"].dtype == pl.Int64
    assert result["target"].to_list() == [0, 1, 0, 1]

    # Check mapping logic (alphabetical or order of appearance depends on polars unique)
    # Default polars unique maintains a stable but internal order
    assert result["cat"].n_unique() == 3
    assert result["num"].n_unique() == 3

def test_ordinal_encoder_inverse(sample_df):
    encoder = OrdinalEncoder(include=["cat"])
    encoded = encoder.fit_transform(sample_df)
    decoded = encoder.inverse_transform(encoded).collect()

    assert decoded["cat"].to_list() == sample_df["cat"].to_list()
    assert decoded["cat"].dtype == pl.String

def test_ordinal_encoder_exclude(sample_df):
    # Process all except "target"
    encoder = OrdinalEncoder(include=None, exclude=["target"])
    result = encoder.fit_transform(sample_df).collect()

    assert result["cat"].dtype == pl.Int64
    assert result["num"].dtype == pl.Int64
    assert result["target"].dtype == pl.Int64
    # Ensure target values didn't change (weren't encoded)
    assert result["target"].to_list() == [0, 1, 0, 1]

def test_ordinal_encoder_allow_unknown_false(sample_df):
    encoder = OrdinalEncoder(include=["cat"], allow_unknown=False)
    encoder.fit(sample_df)

    new_data = pl.DataFrame({"cat": ["z"]}) # "z" was not in fit

    with pytest.raises(Exception): # Polars raises ComputeError
        encoder.transform(new_data).collect()

def test_ordinal_encoder_allow_unknown_true(sample_df):
    encoder = OrdinalEncoder(include=["cat"], allow_unknown=True)
    encoder.fit(sample_df)

    new_data = pl.DataFrame({"cat": ["a", "z"]})
    result = encoder.transform(new_data).collect()

    # "a" should be mapped, "z" should be null
    assert result["cat"][0] is not None
    assert result["cat"][1] is None

def test_ordinal_encoder_pandas_input():
    df_pd = pd.DataFrame({"a": ["x", "y", "x"]})
    encoder = OrdinalEncoder(include=["a"])

    # Should work because of to_lazyframe utility
    result = encoder.fit_transform(df_pd).collect()
    assert isinstance(result, pl.DataFrame)
    assert result["a"].dtype == pl.Int64

def test_with_columns_nonstrict_behavior(sample_df):
    """Verify that if a column is missing during transform, it doesn't crash"""
    encoder = OrdinalEncoder(include=["cat", "num"])
    encoder.fit(sample_df)

    # Dataframe missing the 'num' column
    partial_df = pl.DataFrame({"cat": ["a", "b"]})
    result = encoder.transform(partial_df).collect()

    assert "cat" in result.columns
    assert "num" not in result.columns
    assert result["cat"].dtype == pl.Int64

def test_order_consistency(sample_df):
    """Ensure that the integer mapping is consistent between fit and transform"""
    encoder = OrdinalEncoder(include=["cat"])
    encoder.fit(sample_df)

    val_a = encoder.transform(pl.DataFrame({"cat": ["a"]})).collect()["cat"][0]
    val_b = encoder.transform(pl.DataFrame({"cat": ["b"]})).collect()["cat"][0]

    assert val_a != val_b

    # Second transform should yield same results
    assert encoder.transform(pl.DataFrame({"cat": ["a"]})).collect()["cat"][0] == val_a

def test_ordinal_round_trip(sample_df):
    """Verifies that fit -> transform -> inverse_transform returns the exact original frame."""
    encoder = OrdinalEncoder(include=["cat", "num"])

    encoded = encoder.fit_transform(sample_df)
    decoded = encoder.inverse_transform(encoded).collect()

    # assert_frame_equal checks schema, types, and values
    assert_frame_equal(decoded, sample_df)

def test_ordinal_exclude_logic(sample_df):
    """Verifies that excluded columns remain completely untouched (including type)."""
    # Exclude 'target' and 'num', only encode 'cat'
    encoder = OrdinalEncoder(include=None, exclude=["target", "cat"])

    transformed = encoder.fit_transform(sample_df).collect()

    # Manually create expected: only 'cat' should be Int64
    # We don't check exact integer values here because unique() order can vary,
    # but we check that the other columns are identical to original.
    assert transformed["num"].dtype == pl.Int64
    assert transformed["target"].dtype == pl.Int64
    assert transformed["cat"].dtype == pl.String

    # Verify values of non-transformed columns haven't changed
    assert_frame_equal(transformed.select("target", "cat"), sample_df.select("target", "cat"))

def test_ordinal_consistency_with_mapping(sample_df):
    """Verifies that the encoding is deterministic and follows the internal maps."""
    encoder = OrdinalEncoder(include=["cat"])
    encoder.fit(sample_df)

    # Get the integer assigned to 'b' from the internal state
    mapping = encoder.maps_["cat"]
    val_b = mapping["b"]

    test_df = pl.DataFrame({"cat": ["b", "b", "a"]})
    transformed = encoder.transform(test_df).collect()

    expected = pl.DataFrame({
        "cat": [val_b, val_b, mapping["a"]]
    }).with_columns(pl.col("cat").cast(pl.Int64))

    assert_frame_equal(transformed, expected)

def test_ordinal_allow_unknown(sample_df):
    """Verifies that unknown values become null when allow_unknown=True."""
    encoder = OrdinalEncoder(include=["cat"], allow_unknown=True)
    encoder.fit(sample_df)

    # 'z' was not in the original fit data
    new_data = pl.DataFrame({"cat": ["a", "z"]})
    transformed = encoder.transform(new_data).collect()

    mapping = encoder.maps_["cat"]
    expected = pl.DataFrame({
        "cat": [mapping["a"], None]
    }).with_columns(pl.col("cat").cast(pl.Int64))

    assert_frame_equal(transformed, expected)

def test_with_columns_nonstrict_missing_col(sample_df):
    """Tests the 'nonstrict' utility: if a column is missing in transform, it shouldn't crash."""
    encoder = OrdinalEncoder(include=["cat", "num"])
    encoder.fit(sample_df)

    # Frame missing 'num'
    partial_df = pl.DataFrame({"cat": ["a", "b"]})
    transformed = encoder.transform(partial_df).collect()

    # Should only have 'cat' encoded
    mapping = encoder.maps_["cat"]
    expected = pl.DataFrame({
        "cat": [mapping["a"], mapping["b"]]
    }).with_columns(pl.col("cat").cast(pl.Int64))

    assert_frame_equal(transformed, expected)

def test_ordinal_propagate_nulls_false():
    """Verify how the encoder handles existing nulls in the data."""
    df = pl.DataFrame({"a": ["x", None, "y"]})
    encoder = OrdinalEncoder(include="a", propagate_nulls=False)

    # This should treat None as just another category
    encoded = encoder.fit_transform(df)
    decoded = encoder.inverse_transform(encoded).collect()

    assert_frame_equal(decoded, df)
    # Ensure that None was actually encoded to an integer and not left as a Polars Null
    assert encoded.collect()["a"].null_count() == 0

def test_ordinal_encoder_basic_2():
    df = pl.DataFrame({
        "cat": ["a", "b", "a", "c"],
        "num": [1, 2, 1, 3]
    })

    # We sort in the test expectation assuming we might add .sort()
    # or that Polars returns them in a specific order.
    encoder = OrdinalEncoder(include=["cat"])
    encoder.fit(df)

    transformed = encoder.transform(df).collect()

    assert transformed["cat"].dtype == pl.Int64
    assert transformed["cat"].n_unique() == 3
    assert transformed["num"].dtype == pl.Int64  # Untouched column

    # Check inverse
    inverse = encoder.inverse_transform(transformed).collect()
    assert_frame_equal(df, inverse)

def test_ordinal_encoder_include_exclude():
    df = pl.DataFrame({
        "a": ["x", "y"],
        "b": ["z", "w"],
        "c": [1, 2]
    })

    # Include only 'a'
    enc = OrdinalEncoder(include=["a"]).fit(df)
    res = enc.transform(df).collect()
    assert res["a"].dtype == pl.Int64
    assert res["b"].dtype == pl.String

    # Exclude 'a'
    enc = OrdinalEncoder(include=None, exclude=["a"]).fit(df)
    res = enc.transform(df).collect()
    assert res["a"].dtype == pl.String
    assert res["b"].dtype == pl.Int64

def test_ordinal_encoder_propagate_nulls_true():
    df = pl.DataFrame({"a": ["x", None, "y"]})

    encoder = OrdinalEncoder(include=["a"], propagate_nulls=True)
    encoder.fit(df)
    res = encoder.transform(df).collect()

    # Null should remain null
    assert res["a"][1] is None
    assert res["a"].null_count() == 1

    # Round trip
    inverse = encoder.inverse_transform(res).collect()
    assert_frame_equal(df, inverse)

def test_ordinal_encoder_propagate_nulls_false():
    df = pl.DataFrame({"a": ["x", None, "y"]})

    # When propagate_nulls is False, null is treated as a category
    encoder = OrdinalEncoder(include=["a"], propagate_nulls=False)
    encoder.fit(df)
    res = encoder.transform(df).collect()

    # Null should now be an integer
    assert res["a"].null_count() == 0
    assert isinstance(res["a"][1], int)

    # Round trip
    inverse = encoder.inverse_transform(res).collect()
    assert_frame_equal(df, inverse)

def test_ordinal_encoder_unknown_values():
    train = pl.DataFrame({"a": ["x", "y"]})
    test = pl.DataFrame({"a": ["x", "z"]}) # 'z' is new

    # 1. Error case (allow_unknown=False)
    encoder = OrdinalEncoder(include=["a"], allow_unknown=False)
    encoder.fit(train)
    with pytest.raises(Exception): # replace_strict raises error
        encoder.transform(test).collect()

    # 2. Map to null case (allow_unknown=True)
    encoder_ok = OrdinalEncoder(include=["a"], allow_unknown=True)
    encoder_ok.fit(train)
    res = encoder_ok.transform(test).collect()

    assert res["a"][0] is not None
    assert res["a"][1] is None # 'z' became null

def test_ordinal_encoder_lazy_input():
    df = pl.DataFrame({"a": ["x", "y"]}).lazy()
    encoder = OrdinalEncoder(include=["a"])

    # Ensure fit and transform handle LazyFrames
    encoder.fit(df)
    res = encoder.transform(df)

    assert isinstance(res, pl.LazyFrame)
    assert res.collect()["a"].dtype == pl.Int64

def test_ordinal_encoder_non_existent_col_in_transform():
    # Tests with_columns_nonstrict
    train = pl.DataFrame({"a": ["x", "y"]})
    test = pl.DataFrame({"b": [1, 2]}) # Column 'a' missing

    encoder = OrdinalEncoder(include=["a"]).fit(train)
    # Should not crash, just returns test as is because 'a' isn't there
    res = encoder.transform(test).collect()
    assert_frame_equal(res, test)



def test_map_encoder_basic():
    df = pl.DataFrame({
        "cat": ["a", "b", "a", "c"],
        "num": [10, 20, 10, 30]
    })

    mapping = {
        "cat": {"a": 0, "b": 1, "c": 2},
        "num": {10: 100, 20: 200, 30: 300}
    }

    encoder = MapEncoder(map=mapping)
    encoded_df = encoder.fit_transform(df).collect()

    expected = pl.DataFrame({
        "cat": [0, 1, 0, 2],
        "num": [100, 200, 100, 300]
    }).with_columns(pl.all().cast(pl.Int64))

    assert_frame_equal(encoded_df, expected)

def test_map_encoder_allow_nulls():
    df = pl.DataFrame({
        "a": ["x", None, "y"]
    })

    # allow_nulls=True is default
    encoder = MapEncoder(map={"a": {"x": 1, "y": 2}})
    encoded_df = encoder.fit_transform(df).collect()

    expected = pl.DataFrame({
        "a": [1, None, 2]
    }).with_columns(pl.col("a").cast(pl.Int64))

    assert_frame_equal(encoded_df, expected)


def test_map_encoder_allow_unknown():
    df = pl.DataFrame({
        "a": ["x", "unknown_val"]
    })

    # allow_unknown=True maps missing keys to None
    encoder = MapEncoder(map={"a": {"x": 1}}, allow_unknown=True)
    encoded_df = encoder.fit_transform(df).collect()

    expected = pl.DataFrame({
        "a": [1, None]
    }).with_columns(pl.col("a").cast(pl.Int64))

    assert_frame_equal(encoded_df, expected)

def test_map_encoder_inverse_transform():
    df = pl.DataFrame({
        "a": ["x", "y", "x"]
    })

    mapping = {"a": {"x": 10, "y": 20}}
    encoder = MapEncoder(map=mapping)

    encoded = encoder.fit_transform(df)
    decoded = encoder.inverse_transform(encoded).collect()

    assert_frame_equal(decoded, df)

def test_map_encoder_raises_on_missing_key():
    """Test that it only transforms columns present in the map and the dataframe."""
    df = pl.DataFrame({
        "a": ["x", "y"],
        "b": [1, 2]
    })

    # Mapping contains "c" which isn't in df
    mapping = {
        "a": {"x": 0, "y": 1},
        "c": {"foo": 100}
    }

    encoder = MapEncoder(map=mapping)

    with pytest.raises(KeyError):
        encoder.fit_transform(df).collect()

def test_map_encoder_partial_columns():
    """Test that it only transforms columns present in the map and the dataframe."""
    df = pl.DataFrame({
        "a": ["x", "y"],
        "b": [1, 2]
    })

    # Mapping contains "c" which isn't in df, and df contains "b" which isn't in map
    mapping = {
        "a": {"x": 0, "y": 1},
    }

    encoder = MapEncoder(map=mapping)
    result = encoder.fit_transform(df).collect()

    expected = pl.DataFrame({
        "a": [0, 1],
        "b": [1, 2]
    }).with_columns(pl.col("a").cast(pl.Int64))

    assert_frame_equal(result, expected)

def test_map_encoder_input_mutation():
    """Verify that the encoder does not mutate the user-provided dictionary."""
    user_map = {"a": {"x": 1}}
    encoder = MapEncoder(map=user_map)

    encoder.fit(pl.DataFrame({"a": ["x"]}))

    # If fixed, None should NOT be in user_map["a"]
    assert None not in user_map["a"]