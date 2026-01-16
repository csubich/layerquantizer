import numpy as np
import pytest
from layerquantizer.layerquantizer import (
    quantizer, dequantizer, negabinary, binanegary, lorenzo2d, unlorenzo2d, LayerQuantizer
)

def test_quantizer_basic():
    nbits = 8
    n_planes, height, width = 1, 2, 2
    buf = np.array([[[0.0, 10.0], [5.0, 7.5]]], dtype=np.float32)
    plane_min = np.array([0.0], dtype=np.float32)
    plane_max = np.array([10.0], dtype=np.float32)
    
    # MAX_LEVEL = 2^8 - 1 = 255
    # plane_delta = 10.0 - 0.0 = 10.0
    # plane_scale = 255 / 10.0 = 25.5
    # 0.0 -> round(25.5 * 0) = 0
    # 10.0 -> round(25.5 * 10) = 255
    # 5.0 -> round(25.5 * 5) = 128 (round(127.5) can be 128 or 127 depending on rounding mode, 
    # but SPEC says round(plane_scale * (x - plane_min[i])))
    # 7.5 -> round(25.5 * 7.5) = round(191.25) = 191
    
    quantized = quantizer(buf, nbits, plane_min, plane_max)
    expected = np.array([[[0, 255], [128, 191]]], dtype=np.int32)
    np.testing.assert_array_equal(quantized, expected)

def test_dequantizer_basic():
    nbits = 8
    n_planes, height, width = 1, 2, 2
    quantized = np.array([[[0, 255], [128, 191]]], dtype=np.int32)
    plane_min = np.array([0.0], dtype=np.float32)
    plane_max = np.array([10.0], dtype=np.float32)
    
    # MAX_LEVEL = 255
    # delta = (10.0 - 0.0) / 255
    # 0 -> 0 * delta + 0 = 0
    # 255 -> 255 * delta + 0 = 10.0
    # 128 -> 128 * 10/255 = 5.0196...
    # 191 -> 191 * 10/255 = 7.4901...
    
    dequantized = dequantizer(quantized, nbits, plane_min, plane_max)
    delta = 10.0 / 255
    expected = np.array([[[0.0, 10.0], [128 * delta, 191 * delta]]], dtype=np.float32)
    np.testing.assert_allclose(dequantized, expected)

def test_quantizer_nan_inf():
    nbits = 16
    nan_sigil = 2**nbits
    buf = np.array([[[np.nan, np.inf, -np.inf, 1.0]]], dtype=np.float32)
    plane_min = np.array([0.0], dtype=np.float32)
    plane_max = np.array([1.0], dtype=np.float32)
    
    quantized = quantizer(buf, nbits, plane_min, plane_max)
    assert quantized[0, 0, 0] == nan_sigil
    assert quantized[0, 0, 1] == nan_sigil
    assert quantized[0, 0, 2] == nan_sigil
    assert quantized[0, 0, 3] == (2**nbits - 1)
    
    dequantized = dequantizer(quantized, nbits, plane_min, plane_max)
    assert np.isnan(dequantized[0, 0, 0])
    assert np.isnan(dequantized[0, 0, 1])
    assert np.isnan(dequantized[0, 0, 2])
    assert dequantized[0, 0, 3] == 1.0

def test_quantizer_zero_delta():
    # If plane_delta <= 0, plane_delta is treated as 1
    nbits = 16
    buf = np.array([[[5.0]]], dtype=np.float32)
    plane_min = np.array([5.0], dtype=np.float32)
    plane_max = np.array([5.0], dtype=np.float32)
    
    quantized = quantizer(buf, nbits, plane_min, plane_max)
    # plane_delta = 0 -> 1. plane_scale = (2^16-1)/1 = 65535.
    # round(65535 * (5.0 - 5.0)) = 0
    assert quantized[0, 0, 0] == 0

def test_negabinary_roundtrip():
    # Test with a variety of integers
    vals = np.array([-1000, -1, 0, 1, 1000, 2**30, -2**30], dtype=np.int32)
    encoded = negabinary(vals)
    assert encoded.dtype == np.uint32
    decoded = binanegary(encoded)
    np.testing.assert_array_equal(vals, decoded)

def test_lorenzo_2d_roundtrip():
    # Test 3D array (treated as 2D planes)
    shape = (2, 4, 4)
    data = np.random.randint(-100, 100, shape, dtype=np.int32)
    encoded = lorenzo2d(data)
    decoded = unlorenzo2d(encoded)
    np.testing.assert_array_equal(data, decoded)

def test_lorenzo_1d_cases():
    # Small dimensions
    shape = (1, 1, 5)
    data = np.random.randint(-100, 100, shape, dtype=np.int32)
    encoded = lorenzo2d(data)
    decoded = unlorenzo2d(encoded)
    np.testing.assert_array_equal(data, decoded)
    
    shape = (1, 5, 1)
    data = np.random.randint(-100, 100, shape, dtype=np.int32)
    encoded = lorenzo2d(data)
    decoded = unlorenzo2d(encoded)
    np.testing.assert_array_equal(data, decoded)

def test_layerquantizer_roundtrip():
    codec = LayerQuantizer(nbits=12, transform="Lorenzo")
    data = np.random.uniform(-10, 10, (2, 8, 8)).astype(np.float32)
    
    encoded = codec.encode(data)
    decoded = codec.decode(encoded)
    
    # Reshape decoded to match original (it might come back flattened or with different shape if not careful)
    decoded = decoded.reshape(data.shape)
    
    # Precision check
    # Max error should be roughly delta / 2
    # delta = (max - min) / (2^12 - 1)
    # Here range is ~20, so delta is ~20/4095 ~ 0.0048
    np.testing.assert_allclose(data, decoded, atol=0.01)

def test_layerquantizer_no_transform():
    codec = LayerQuantizer(nbits=12, transform=None)
    data = np.random.uniform(-10, 10, (2, 8, 8)).astype(np.float32)
    
    encoded = codec.encode(data)
    decoded = codec.decode(encoded)
    decoded = decoded.reshape(data.shape)
    
    np.testing.assert_allclose(data, decoded, atol=0.01)

def test_layerquantizer_all_nan_plane():
    codec = LayerQuantizer(nbits=16)
    data = np.full((1, 4, 4), np.nan, dtype=np.float32)
    
    encoded = codec.encode(data)
    decoded = codec.decode(encoded)
    assert np.all(np.isnan(decoded))

def test_layerquantizer_constant_zero():
    codec = LayerQuantizer(nbits=16)
    data = np.zeros((1, 4, 4), dtype=np.float32)
    
    encoded = codec.encode(data)
    decoded = codec.decode(encoded)
    np.testing.assert_array_equal(data, decoded.reshape(data.shape))

def test_layerquantizer_constant_nonzero():
    codec = LayerQuantizer(nbits=16)
    val = 42.0
    data = np.full((1, 4, 4), val, dtype=np.float32)
    
    encoded = codec.encode(data)
    decoded = codec.decode(encoded)
    np.testing.assert_allclose(data, decoded.reshape(data.shape), atol=1e-5)

@pytest.mark.parametrize("nbits", [8, 12, 16, 23])
@pytest.mark.parametrize("transform", ["Lorenzo", None])
@pytest.mark.parametrize("pow2_range", [True, False])
def test_layerquantizer_comprehensive(nbits, transform, pow2_range):
    codec = LayerQuantizer(nbits=nbits, transform=transform, pow2_range=pow2_range)
    
    # Test a variety of ranges, especially those near powers of 2
    # Ranges: [0.5, 0.99, 1.0, 1.01, 1.99, 2.0, 2.01, 10.0, 100.0]
    ranges = [0.5, 0.99, 1.0, 1.01, 1.99, 2.0, 2.01, 10.0, 100.0]
    for r in ranges:
        data = np.array([[[0.0, r]]], dtype=np.float32)
        encoded = codec.encode(data)
        decoded = codec.decode(encoded).reshape(data.shape)
        
        # Basic round-trip validation
        # The reconstructed 'r' should be very close to original 'r' if nbits is high enough
        # Error bound is roughly (expanded_range / (2^nbits - 1))
        # For pow2_range=True, expanded_range can be up to 2x the original range.
        
        # Determine expected precision
        plane_delta = r
        if pow2_range:
            # Replicate spec logic for expanded range
            scale_factor = (2**nbits) / (2**nbits - 1)
            plane_delta = 2**(1 + np.floor(np.log2(r * scale_factor)))
            
        max_error = plane_delta / (2**nbits - 1)
        np.testing.assert_allclose(data, decoded, atol=max_error + 1e-7)
        
        # Ensure that the decoded value is within the range [0, r] or slightly beyond if r was plane_max
        # (Though with quantization it should be exactly on a grid point)
        assert np.all(decoded >= -1e-7)
        # For pow2_range=False, r is plane_max, so it should be exactly r.
        # For pow2_range=True, r is <= plane_max.

def test_layerquantizer_fallback_high_nbits():
    # SPEC: If nbits > 24, falls back to raw Blosc
    # Update: SPEC now says decode should handle this correctly.
    codec = LayerQuantizer(nbits=25)
    data = np.random.rand(1, 4, 4).astype(np.float32)
    
    encoded = codec.encode(data)
    decoded = codec.decode(encoded)
    
    # Should be exact
    np.testing.assert_array_equal(data, decoded.reshape(data.shape))

def test_layerquantizer_pow2_range():
    # Test that pow2_range expansion works and still yields a valid round-trip
    codec = LayerQuantizer(nbits=16, pow2_range=True)
    data = np.array([[[0.0, 3.0]]], dtype=np.float32) # range 3.0
    # Next power of 2 for range is 4.0 (actually SPEC says 2^(1 + floor(log2(delta))))
    # log2(3) = 1.58 -> floor = 1 -> 2^(1+1) = 4.0.
    
    encoded = codec.encode(data)
    decoded = codec.decode(encoded)
    np.testing.assert_allclose(data, decoded.reshape(data.shape), atol=1e-4)

def test_layerquantizer_large_range():
    codec = LayerQuantizer(nbits=16)
    data = np.array([[[-1e30, 1e30]]], dtype=np.float32)
    
    encoded = codec.encode(data)
    decoded = codec.decode(encoded)
    # Check that we don't crash and get some reasonable values (quantization will be very coarse)
    assert not np.any(np.isnan(decoded))
    np.testing.assert_allclose(data, decoded.reshape(data.shape), rtol=1e-3)

def test_denormal_values():
    # Denormals are small.
    codec = LayerQuantizer(nbits=16)
    denormal = np.array([[[1e-40, 2e-40]]], dtype=np.float32)
    
    encoded = codec.encode(denormal)
    decoded = codec.decode(encoded)
    # They might be flushed to zero, but should at least not crash.
    # If not flushed, they should round-trip approximately.
    # Since range is ~1e-40, and min is 1e-40, plane_delta = 1e-40.
    # Adjusted max will be non-zero.
    assert not np.any(np.isnan(decoded))

def test_dtype_validation():
    codec = LayerQuantizer(nbits=16)
    data = np.zeros((2, 2, 2), dtype=np.float64)
    with pytest.raises(Exception): # SPEC says it must have dtype=np.float32
        codec.encode(data)
