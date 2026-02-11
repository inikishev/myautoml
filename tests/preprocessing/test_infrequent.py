import polars as pl
import pytest
from polars.testing import assert_frame_equal

from myautoml.polars_transformers.infrequent import MergeInfrequent

# Assuming the classes are in the correct import path
# from myautoml.preprocessing.infrequent import MergeInfrequent

def test_merge_infrequent_min_frequency_int():
    df = pl.DataFrame({
        "feature": ["A", "A", "A", "B", "B", "C", "D"], # A:3, B:2, C:1, D:1
    })

    # Threshold = 2: C and D should be merged
    transformer = MergeInfrequent(include="feature", min_frequency=2)
    result = transformer.fit_transform(df).collect()

    expected = pl.DataFrame({
        "feature": ["A", "A", "A", "B", "B", "__infrequent", "__infrequent"]
    })
    assert_frame_equal(result, expected)

def test_merge_infrequent_min_frequency_float():
    df = pl.DataFrame({
        "feature": ["A", "A", "A", "A", "B", "C", "D"], # Total 7. A: ~57%, B: ~14%
    })

    # Threshold = 0.2: B (1/7 = 14%), C, and D should be merged
    transformer = MergeInfrequent(include="feature", min_frequency=0.2)
    result = transformer.fit_transform(df).collect()

    expected = pl.DataFrame({
        "feature": ["A", "A", "A", "A", "__infrequent", "__infrequent", "__infrequent"]
    })
    assert_frame_equal(result, expected)

def test_merge_infrequent_max_categories():
    df = pl.DataFrame({
        "feature": ["A", "A", "A", "B", "B", "C", "D", "E"],
    })
    # Keep top 2 (A and B), merge others
    transformer = MergeInfrequent(include="feature", max_categories=2)
    result = transformer.fit_transform(df).collect()

    expected = pl.DataFrame({
        "feature": ["A", "A", "A", "B", "B", "__infrequent", "__infrequent", "__infrequent"]
    })
    assert_frame_equal(result, expected)

def test_merge_infrequent_propagate_nulls_true():
    df = pl.DataFrame({
        "feature": ["A", "A", "B", None, None]
    })
    # min_freq=2. 'B' is infrequent. Null should remain Null.
    transformer = MergeInfrequent(include="feature", min_frequency=2, propagate_nulls=True)
    result = transformer.fit_transform(df).collect()

    expected = pl.DataFrame({
        "feature": ["A", "A", "__infrequent", None, None]
    })
    assert_frame_equal(result, expected)

def test_merge_infrequent_propagate_nulls_false():
    df = pl.DataFrame({
        "feature": ["A", "A", "B", None]
    })
    # min_freq=2. 'B' (1) and None (1) are infrequent. Both become "__infrequent".
    transformer = MergeInfrequent(include="feature", min_frequency=2, propagate_nulls=False)
    result = transformer.fit_transform(df).collect()
    expected = pl.DataFrame({
        "feature": ["A", "A", "__infrequent", "__infrequent"]
    })
    assert_frame_equal(result, expected)

def test_merge_infrequent_numeric_column():
    # Testing with numeric value to avoid type errors
    df = pl.DataFrame({"feature": [1, 1, 1, 2, 3]})

    transformer = MergeInfrequent(include="feature", min_frequency=2, infrequent_value=-1)
    result = transformer.fit_transform(df).collect()

    expected = pl.DataFrame({"feature": [1, 1, 1, -1, -1]})
    assert_frame_equal(result, expected)

def test_merge_infrequent_include_exclude():
    df = pl.DataFrame({
        "a": ["A", "A", "B"],
        "b": ["X", "Y", "Z"]
    })
    # Only process 'a'
    transformer = MergeInfrequent(include=["a"], min_frequency=2)
    result = transformer.fit_transform(df).collect()

    assert result["a"][2] == "__infrequent"
    assert result["b"][2] == "Z"  # Remains unchanged

def test_merge_infrequent_noop():
    df = pl.DataFrame({"a": ["A", "B"]})
    # No criteria provided
    transformer = MergeInfrequent(include=None)
    result = transformer.fit_transform(df).collect()

    assert_frame_equal(result, df)

def test_merge_infrequent_no_infrequent_found():
    df = pl.DataFrame({"a": ["A", "A", "B", "B"]})
    # All meet the threshold
    transformer = MergeInfrequent(include="a", min_frequency=2)
    result = transformer.fit_transform(df).collect()

    assert_frame_equal(result, df)