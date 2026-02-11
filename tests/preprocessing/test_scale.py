import pytest
import polars as pl
import polars.testing as pl_testing
import numpy as np
from myautoml.polars_transformers.scale import StandardScaler, MinMaxScaler, SampleNormalizer

@pytest.fixture
def sample_data():
    return pl.DataFrame({
        "a": [1.0, 2.0, 3.0],
        "b": [10.0, 20.0, 30.0],
        "c": ["x", "y", "z"]
    })

class TestStandardScaler:
    def test_standard_scaler_basic(self, sample_data):
        scaler = StandardScaler()
        # Fit and transform
        transformed = scaler.fit_transform(sample_data).collect()

        # Mean of a is 2.0, std is 1.0. (1-2)/1 = -1, (2-2)/1 = 0, (3-2)/1 = 1
        assert transformed["a"].to_list() == [-1.0, 0.0, 1.0]
        # Mean of b is 20.0, std is 10.0. (10-20)/10 = -1, (20-20)/10 = 0, (30-20)/10 = 1
        assert transformed["b"].to_list() == [-1.0, 0.0, 1.0]
        # Column c should be untouched
        assert transformed["c"].to_list() == ["x", "y", "z"]

    def test_standard_scaler_inverse(self, sample_data):
        scaler = StandardScaler()
        transformed = scaler.fit_transform(sample_data)
        inverted = scaler.inverse_transform(transformed).collect()

        pl_testing.assert_frame_equal(inverted, sample_data)

    def test_standard_scaler_include_exclude(self, sample_data):
        # Only scale column 'a'
        scaler = StandardScaler(include=["a"])
        transformed = scaler.fit_transform(sample_data).collect()

        assert transformed["a"].to_list() == [-1.0, 0.0, 1.0]
        assert transformed["b"].to_list() == [10.0, 20.0, 30.0] # Unchanged

    def test_standard_scaler_no_mean_no_std(self, sample_data):
        scaler = StandardScaler(with_mean=False, with_std=False)
        transformed = scaler.fit_transform(sample_data).collect()
        pl_testing.assert_frame_equal(transformed, sample_data)

class TestMinMaxScaler:
    def test_min_max_scaler_basic(self, sample_data):
        scaler = MinMaxScaler()
        transformed = scaler.fit_transform(sample_data).collect()

        # Col a: min 1, max 3 -> [0, 0.5, 1]
        assert transformed["a"].to_list() == [0.0, 0.5, 1.0]
        # Col b: min 10, max 30 -> [0, 0.5, 1]
        assert transformed["b"].to_list() == [0.0, 0.5, 1.0]

    def test_min_max_scaler_range(self, sample_data):
        scaler = MinMaxScaler(feature_range=(-1, 1))
        transformed = scaler.fit_transform(sample_data).collect()

        assert transformed["a"].to_list() == [-1.0, 0.0, 1.0]

    def test_min_max_scaler_clip(self):
        df = pl.DataFrame({"a": [1.0, 2.0, 3.0]})
        scaler = MinMaxScaler(feature_range=(0, 1), clip=True)
        scaler.fit(df)

        # Test with data out of bounds
        test_df = pl.DataFrame({"a": [0.0, 4.0]})
        transformed = scaler.transform(test_df).collect()

        assert transformed["a"].to_list() == [0.0, 1.0]

    def test_min_max_inverse(self, sample_data):
        scaler = MinMaxScaler(feature_range=(5, 10))
        transformed = scaler.fit_transform(sample_data)
        inverted = scaler.inverse_transform(transformed).collect()

        pl_testing.assert_frame_equal(inverted, sample_data)

class TestSampleNormalizer:
    def test_sample_normalizer_basic(self):
        # Row 1: Mean=2, Std=1 -> [-1, 1]
        # Row 2: Mean=20, Std=10 -> [-1, 1]
        df = pl.DataFrame({
            "a": [1.0, 10.0],
            "b": [3.0, 30.0]
        })

        normalizer = SampleNormalizer()
        transformed = normalizer.fit_transform(df).collect()

        # Check row 1
        assert transformed[0, "a"] == pytest.approx(-0.707106, abs=1e-5) # 1 / sqrt(2) approx
        assert transformed[0, "b"] == pytest.approx(0.707106, abs=1e-5)

    def test_sample_normalizer_include(self):
        df = pl.DataFrame({
            "a": [1.0, 2.0],
            "b": [10.0, 20.0],
            "c": [0.0, 0.0]
        })
        # Only normalize across a and b, ignore c
        normalizer = SampleNormalizer(include=["a", "b"])
        transformed = normalizer.fit_transform(df).collect()

        # Column c should remain 0.0
        assert transformed["c"].to_list() == [0.0, 0.0]
        # a and b should be normalized relative to each other per row
        assert transformed[0, "a"] < 0
        assert transformed[0, "b"] > 0

def test_lazy_input_support():
    """Verify that all scalers handle LazyFrames and return LazyFrames."""
    df = pl.DataFrame({"a": [1, 2, 3]}).lazy()

    scalers = [StandardScaler(), MinMaxScaler(), SampleNormalizer()]

    for scaler in scalers:
        result = scaler.fit_transform(df)
        assert isinstance(result, pl.LazyFrame)
        # Verify it can be collected
        result.collect()

def test_constant_column_handling():
    """Verify that scalers don't crash on zero variance (division by zero)."""
    df = pl.DataFrame({"a": [1.0, 1.0, 1.0]})

    std_scaler = StandardScaler().fit(df)
    min_max = MinMaxScaler().fit(df)

    # Standard scaler should handle 1e-16 clip from your source code
    # (1-1) / 1e-16 = 0
    assert std_scaler.transform(df).collect()["a"][0] == 0.0
    assert min_max.transform(df).collect()["a"][0] == 0.0