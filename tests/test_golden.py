import numpy as np
from layerquantizer.layerquantizer import LayerQuantizer
import base64

def test_golden_master_reproducibility():
    """Ensure that the on-disk representation is stable."""
    # Seed for reproducibility
    np.random.seed(42)
    data = np.random.randn(2, 4, 4).astype(np.float32)
    
    # Standard codec settings
    codec = LayerQuantizer(nbits=16, transform="Lorenzo", blosc_cname="zstd", blosc_clevel=5)
    
    # Encode
    encoded = codec.encode(data)
    
    # The 'golden' base64 representation of the encoded data at commit 872791e
    # If the on-disk format changes, this will fail.
    golden_b64 = (
        "AgGRBJwAAACcAAAAmAAAABQAAACAAAAAKLUv/SCcvQMAlAYCBAReJqV0+rXsfrKfU1XAc6iqeiucp3L/he/td6XWD1eGKu8xrLEAAADmXiMX"
        "8dZOQN9zX4QzXnZMdGioqmC44SnGFIv3p8ubPAwlX1kAAAD0tsrtAQADAAADAQEAAQAAAQMAv78/PwAEALimPomhBHoRIAo="
    )
    
    current_b64 = base64.b64encode(encoded).decode('ascii')
    
    assert current_b64 == golden_b64, f"Encoded representation mismatch!\nExpected: {golden_b64}\nActual:   {current_b64}"
    
    # Also verify round-trip from the golden representation
    golden_bytes = base64.b64decode(golden_b64)
    decoded = codec.decode(golden_bytes).reshape(data.shape)
    
    # Basic round-trip validation
    np.testing.assert_allclose(data, decoded, atol=1e-3)

if __name__ == "__main__":
    # If run directly, generate the golden string
    np.random.seed(42)
    data = np.random.randn(2, 4, 4).astype(np.float32)
    codec = LayerQuantizer(nbits=16, transform="Lorenzo", blosc_cname="zstd", blosc_clevel=5)
    encoded = codec.encode(data)
    print(base64.b64encode(encoded).decode('ascii'))
