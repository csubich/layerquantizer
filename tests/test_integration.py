import numpy as np
import xarray as xr
import zarr
import pytest
from layerquantizer.layerquantizer import LayerQuantizer

def test_xarray_zarr_integration():
    # 1. Create synthetic data
    # Shape: (time, level, lat, lon) or similar
    # LayerQuantizer works on the last two dimensions as planes, 
    # and treats everything before that as the "planes" dimension.
    # For a 3D array (nplanes, height, width), it works as expected.
    # For a 4D array, it will be reshaped to (dim0*dim1, dim2, dim3).
    
    ntime, nlevel, nlat, nlon = 2, 3, 4, 5
    data = np.random.rand(ntime, nlevel, nlat, nlon).astype(np.float32)
    
    ds = xr.Dataset(
        data_vars={
            "test_var": (("time", "level", "lat", "lon"), data)
        },
        coords={
            "time": np.arange(ntime),
            "level": np.arange(nlevel),
            "lat": np.arange(nlat),
            "lon": np.arange(nlon),
        }
    )
    
    # 2. Define compressor
    lq = LayerQuantizer(nbits=16, transform="Lorenzo")
    
    # 3. Use MemoryStore for zarr
    store = zarr.MemoryStore()
    
    # 4. Write to zarr with LayerQuantizer
    encoding = {"test_var": {"compressor": lq}}
    ds.to_zarr(store, encoding=encoding, zarr_format=2, compute=True)
    
    # 5. Read back
    ds_read = xr.open_zarr(store, zarr_format=2)
    data_read = ds_read.test_var.values
    
    # 6. Verify
    # Since it's lossy, we check with assert_allclose
    # Range is ~1.0, nbits=16 -> delta ~ 1/65535 ~ 1.5e-5
    np.testing.assert_allclose(data, data_read, atol=1e-4)

def test_xarray_zarr_high_nbits_fallback():
    # Test fallback path (nbits > 24) through xarray/zarr
    ntime, nlat, nlon = 1, 4, 4
    data = np.random.rand(ntime, nlat, nlon).astype(np.float32)
    ds = xr.Dataset({"v": (("time", "lat", "lon"), data)})
    
    lq = LayerQuantizer(nbits=25)
    store = zarr.MemoryStore()
    ds.to_zarr(store, encoding={"v": {"compressor": lq}}, zarr_format=2)
    
    ds_read = xr.open_zarr(store, zarr_format=2)
    # Should be exact
    np.testing.assert_array_equal(data, ds_read.v.values)

def test_xarray_zarr_nan_handling():
    ntime, nlat, nlon = 1, 4, 4
    data = np.random.rand(ntime, nlat, nlon).astype(np.float32)
    data[0, 0, 0] = np.nan
    
    ds = xr.Dataset({"v": (("time", "lat", "lon"), data)})
    lq = LayerQuantizer(nbits=16)
    store = zarr.MemoryStore()
    ds.to_zarr(store, encoding={"v": {"compressor": lq}}, zarr_format=2)
    
    ds_read = xr.open_zarr(store, zarr_format=2)
    data_read = ds_read.v.values
    
    assert np.isnan(data_read[0, 0, 0])
    np.testing.assert_allclose(data[0, 0, 1:], data_read[0, 0, 1:], atol=1e-4)

def test_xarray_zarr_pow2_range():
    ntime, nlat, nlon = 1, 4, 4
    # Range [0, 3] -> expanded to 4
    data = np.array([[[0.0, 1.0, 2.0, 3.0]] * 4], dtype=np.float32)
    
    ds = xr.Dataset({"v": (("time", "lat", "lon"), data)})
    lq = LayerQuantizer(nbits=16, pow2_range=True)
    store = zarr.MemoryStore()
    ds.to_zarr(store, encoding={"v": {"compressor": lq}}, zarr_format=2)
    
    ds_read = xr.open_zarr(store, zarr_format=2)
    np.testing.assert_allclose(data, ds_read.v.values, atol=1e-4)
