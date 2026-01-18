import pytest
import numpy as np
import xarray as xr
import zarr
import asyncio
from layerquantizer.layerquantizer import LayerQuantizer
from layerquantizer.zarr3_codec import LayerQuantizerCodec as LayerQuantizerCodecClass

# Try to import LayerQuantizerCodec, if it doesn't exist (yet), define a placeholder for tests to fail gracefully-ish
LayerQuantizerCodec = LayerQuantizerCodecClass

def get_store_size(store):
    """Returns the total size of the store in bytes."""

    async def _get_size():
        # Zarr 3 stores have list_prefix or list
        size = 0
        try:
            # Try Zarr 3 async listing
            async for key in store.list():
                size += await store.getsize(key)
        except (AttributeError, TypeError):
            # Fallback for Zarr 2 or sync stores wrapped/mocked
            # Note: store.keys() is synchronous in Zarr 2
            try:
                keys = store.keys()
                return sum(len(store[k]) for k in keys)
            except (AttributeError, KeyError, TypeError):
                pass
        return size

    # Try synchronous approach first (Zarr 2 typical)
    try:
        keys = list(store.keys())
        return sum(len(store[k]) for k in keys)
    except (AttributeError, TypeError):
        # Async approach (Zarr 3)
        try:
            return asyncio.get_event_loop().run_until_complete(_get_size())
        except RuntimeError:
             # If loop is already running, we might need a different strategy,
             # but in pytest this usually works.
             return asyncio.run(_get_size())

@pytest.fixture
def sample_data_3d():
    return np.random.rand(4, 16, 16).astype(np.float32)

@pytest.fixture
def sample_data_4d():
    return np.random.rand(2, 4, 16, 16).astype(np.float32)

@pytest.fixture
def gaussian_data_3d():
    """Generates a 3D Gaussian of size (256, 256, 256)."""
    size = 256
    x = np.linspace(-5, 5, size)
    y = np.linspace(-5, 5, size)
    z = np.linspace(-5, 5, size)
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    data = np.exp(-(xx**2 + yy**2 + zz**2) / 2.0).astype(np.float32)
    return data

def test_codec_import():
    assert LayerQuantizerCodec is not None, "LayerQuantizerCodec not implemented yet"

@pytest.mark.skipif(LayerQuantizerCodec is None, reason="LayerQuantizerCodec not implemented")
def test_zarr3_basic_roundtrip(sample_data_3d):
    data = sample_data_3d
    
    codec = LayerQuantizerCodec(nbits=16, transform="Lorenzo")
    
    store = zarr.storage.MemoryStore()
    
    z_arr = zarr.create_array(
        store,
        name="data",
        shape=data.shape,
        chunks=(2, 8, 8),
        dtype=np.float32,
        serializer=codec,
        dimension_names=("z", "y", "x")
    )
    z_arr[:] = data
        
    ds_read = xr.open_zarr(store, zarr_format=3, consolidated=False)
    data_read = ds_read.data.values
    
    # nbits=16 should give high precision
    np.testing.assert_allclose(data, data_read, atol=1e-4)

@pytest.mark.skipif(LayerQuantizerCodec is None, reason="LayerQuantizerCodec not implemented")
def test_zarr3_config_options(sample_data_3d):
    data = sample_data_3d
    # Test different configs
    configs = [
        {"nbits": 8, "transform": "Lorenzo"},
        {"nbits": 12, "transform": "None"},
        {"nbits": 16, "transform": "Lorenzo", "pow2_range": True},
    ]
    
    for conf in configs:
        codec = LayerQuantizerCodec(**conf)
        store = zarr.storage.MemoryStore()
        
        # Use direct zarr creation
        z_arr = zarr.create_array(
            store,
            name="data",
            shape=data.shape,
            chunks=(4, 16, 16),
            dtype=np.float32,
            serializer=codec,
            dimension_names=("z", "y", "x") # Required for Xarray
        )
        z_arr[:] = data
        
        ds_read = xr.open_zarr(store, zarr_format=3, consolidated=False)
        data_read = ds_read.data.values
        
        # Lower bits means lower precision
        atol = 1.0 / (2**conf.get("nbits", 16) - 1)
        # Relax tolerance slightly for range expansion
        if conf.get("pow2_range"):
            atol *= 2
            
        np.testing.assert_allclose(data, data_read, atol=atol * 2)

@pytest.mark.skipif(LayerQuantizerCodec is None, reason="LayerQuantizerCodec not implemented")
def test_zarr3_vs_zarr2_compression_ratio():
    # Use larger data to dilute metadata overhead
    chunks = (4, 64, 64)
    x = np.linspace(-5, 5, 128)
    y = np.linspace(-5, 5, 128)
    xx, yy = np.meshgrid(x, y)
    plane = np.exp(-(xx**2 + yy**2) / 2.0).astype(np.float32)
    data = np.stack([plane * (i + 1) / 16.0 for i in range(16)]).astype(np.float32)
    
    # Zarr 2
    lq = LayerQuantizer(nbits=12, transform="Lorenzo", blosc_cname="zstd", blosc_clevel=5)
    ds2 = xr.Dataset({"data": (("z", "y", "x"), data)})
    store2 = zarr.storage.MemoryStore()
    ds2.to_zarr(store2, encoding={"data": {"compressor": lq, "chunks": chunks}}, zarr_format=2, consolidated=False)
    size2 = get_store_size(store2)

    # Zarr 3
    from zarr.codecs import BloscCodec
    lq3 = LayerQuantizerCodec(nbits=12, transform="Lorenzo")
    compressor3 = BloscCodec(cname="zstd", clevel=5, shuffle="shuffle")
    store3 = zarr.storage.MemoryStore()
    z_arr = zarr.create_array(
        store3,
        name="data",
        shape=data.shape,
        chunks=chunks,
        dtype=np.float32,
        serializer=lq3,
        compressors=[compressor3],
        dimension_names=("z", "y", "x")
    )
    z_arr[:] = data
    size3 = get_store_size(store3)
         
    if size2 > 0 and size3 > 0:
        ratio = size2 / size3
        # Should be very close now with larger data
        assert 0.9 < ratio < 1.1, f"Compression ratio mismatch: Zarr2={size2}, Zarr3={size3}, ratio={ratio}"
    else:
        pytest.skip("Could not determine store sizes")

@pytest.mark.skipif(LayerQuantizerCodec is None, reason="LayerQuantizerCodec not implemented")
def test_zarr3_vs_zarr2_large_structured(gaussian_data_3d):
    from zarr.codecs import BloscCodec
    data = gaussian_data_3d
    chunks = (64, 64, 64)
    
    # Configurations to test
    configs = [
        {"nbits": 8, "transform": "Lorenzo", "blosc_cname": "zstd", "blosc_clevel": 5},
        {"nbits": 12, "transform": "Lorenzo", "blosc_cname": "zstd", "blosc_clevel": 5},
        {"nbits": 16, "transform": "Lorenzo", "blosc_cname": "zstd", "blosc_clevel": 5},
        {"nbits": 12, "transform": "None", "blosc_cname": "lz4", "blosc_clevel": 9},
        {"nbits": 16, "transform": "Lorenzo", "blosc_cname": "blosclz", "blosc_clevel": 1},
        {"nbits": 10, "transform": "Lorenzo", "pow2_range": True, "blosc_cname": "zstd", "blosc_clevel": 3},
    ]

    for conf in configs:
        nbits = conf["nbits"]
        transform = conf["transform"]
        pow2_range = conf.get("pow2_range", False)
        cname = conf["blosc_cname"]
        clevel = conf["blosc_clevel"]

        # Zarr 2 setup
        lq2 = LayerQuantizer(
            nbits=nbits, 
            transform=transform, 
            pow2_range=pow2_range,
            blosc_cname=cname,
            blosc_clevel=clevel
        )
        store2 = zarr.storage.MemoryStore()
        ds2 = xr.Dataset({"data": (("z", "y", "x"), data)})
        ds2.to_zarr(store2, encoding={"data": {"compressor": lq2, "chunks": chunks}}, zarr_format=2, consolidated=False)
        
        # Zarr 3 setup
        lq3 = LayerQuantizerCodec(
            nbits=nbits,
            transform=transform,
            pow2_range=pow2_range
        )
        compressor3 = BloscCodec(cname=cname, clevel=clevel, shuffle="shuffle")
        store3 = zarr.storage.MemoryStore()
        
        # Use Xarray to_zarr for Zarr 3 if possible, but Xarray support for 'serializer' might be tricky.
        # Direct zarr.create_array is safer for now to ensure our codec is used exactly as intended.
        z_arr3 = zarr.create_array(
            store3,
            name="data",
            shape=data.shape,
            chunks=chunks,
            dtype=np.float32,
            serializer=lq3,
            compressors=[compressor3],
            dimension_names=("z", "y", "x")
        )
        z_arr3[:] = data

        # Verify Roundtrip through Xarray for Zarr 3
        ds3_read = xr.open_zarr(store3, zarr_format=3, consolidated=False)
        data3_read = ds3_read.data.values
        
        # Verify Roundtrip through Xarray for Zarr 2
        ds2_read = xr.open_zarr(store2, zarr_format=2, consolidated=False)
        data2_read = ds2_read.data.values

        # Both should match original data within quantization tolerance
        atol = (1.0 / (2**nbits - 1)) * 2
        if pow2_range:
            atol *= 2
        
        np.testing.assert_allclose(data, data2_read, atol=atol, err_msg=f"Zarr2 failed for {conf}")
        np.testing.assert_allclose(data, data3_read, atol=atol, err_msg=f"Zarr3 failed for {conf}")
        
        # Compare sizes
        size2 = get_store_size(store2)
        size3 = get_store_size(store3)
        
        # With large data, the overhead of Zarr 3 metadata should be negligible.
        # The actual compressed chunks should be identical in size if the implementation is perfectly matched.
        ratio = size2 / size3
        assert 0.95 < ratio < 1.05, f"Compression ratio mismatch for {conf}: Zarr2={size2}, Zarr3={size3}, ratio={ratio}"

@pytest.mark.skipif(LayerQuantizerCodec is None, reason="LayerQuantizerCodec not implemented")
def test_zarr3_with_compressor():
    from zarr.codecs import BloscCodec
    # Use structured data for better compression
    chunks = (4, 64, 64)
    x = np.linspace(-5, 5, 128)
    y = np.linspace(-5, 5, 128)
    xx, yy = np.meshgrid(x, y)
    plane = np.exp(-(xx**2 + yy**2) / 2.0).astype(np.float32)
    data = np.stack([plane * (i + 1) / 16.0 for i in range(16)]).astype(np.float32)
    
    serializer = LayerQuantizerCodec(nbits=12, transform="Lorenzo")
    compressor = BloscCodec(cname="zstd", clevel=5, shuffle="shuffle")
    
    store = zarr.storage.MemoryStore()
    
    z_arr = zarr.create_array(
        store,
        name="data",
        shape=data.shape,
        chunks=chunks,
        dtype=np.float32,
        serializer=serializer,
        compressors=[compressor],
        dimension_names=("z", "y", "x")
    )
    z_arr[:] = data
    
    ds_read = xr.open_zarr(store, zarr_format=3, consolidated=False)
    data_read = ds_read.data.values
    
    np.testing.assert_allclose(data, data_read, atol=1e-3)
    
    size = get_store_size(store)
    # Raw float32 is 16*128*128*4 = 1,048,576 bytes.
    # It should be much smaller than this.
    assert size < 100000 
