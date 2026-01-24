import numpy as np
import xarray as xr
import fsspec
import warnings
from typing import List
from pathlib import Path

# Suppress ZarrUserWarning about fsspec not being asynchronous
warnings.filterwarnings("ignore", category=UserWarning, message=".*asynchronous=True.*")

# Constants
VARIABLES_3D = [
    "geopotential",
    "potential_vorticity",
    "specific_humidity",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "vertical_velocity",
]

URL_FINE = "gs://weatherbench2/datasets/era5/1959-2023_01_10-full_37-1h-0p25deg-chunk-1.zarr"
URL_COARSE = "gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-64x32_equiangular_conservative.zarr"

# Use absolute path for caching
PROJECT_ROOT = Path(__file__).parents[2]
CACHE_DIR = PROJECT_ROOT / "data" / "cache"

def clear_cache():
    """Clear the local data cache."""
    import shutil
    if CACHE_DIR.exists():
        shutil.rmtree(CACHE_DIR)
        print(f"Cleared cache at {CACHE_DIR}")
    else:
        print(f"Cache directory {CACHE_DIR} does not exist.")

def get_wb_era5(resolution: str = "coarse", cache: bool = True, **kwargs) -> xr.Dataset:
    """
    Load WeatherBench ERA5 dataset.
    
    Args:
        resolution: "coarse" (5.625deg) or "fine" (0.25deg).
        cache: Whether to cache the dataset locally.
        **kwargs: Additional storage options (e.g., token="anon").
    """
    url = URL_COARSE if resolution == "coarse" else URL_FINE
    
    storage_options = {"token": "anon"}
    storage_options.update(kwargs)
    
    if cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        # Use CachingFileSystem (simplecache)
        protocol = url.split("://")[0]
        path = url.split("://")[-1]
        fs = fsspec.filesystem(
            "simplecache",
            target_protocol=protocol,
            target_options=storage_options,
            cache_storage=str(CACHE_DIR)
        )
        mapper = fs.get_mapper(path)
        ds = xr.open_zarr(mapper, chunks={})
    else:
        ds = xr.open_zarr(url, chunks={}, storage_options=storage_options)
        
    return ds

def sample_times(ds: xr.Dataset, n_time: int = 10, seed: int = 42) -> xr.Dataset:
    """Sample random time slices from the dataset."""
    gen = np.random.default_rng(seed=seed)
    sel_times = np.sort(gen.choice(ds.time.data, size=n_time, replace=False))
    return ds.sel(time=sel_times)

async def get_store_size(store) -> int:
    """Get the total size of a Zarr storage object in bytes."""
    # Works for both Zarr 2 and Zarr 3 if 'store' follows the interface
    if hasattr(store, "list"):
        # Zarr 2 style or async Zarr 3
        try:
            # Try Zarr 2 style first
            return sum(store.getsize(k) for k in store.list())
        except (TypeError, AttributeError):
            # Try Zarr 3 / async style
            size = 0
            async for k in store.list():
                size += await store.getsize(k)
            return size
    return 0

def format_bytes(n_bytes: int) -> str:
    """Format bytes to MiB."""
    return f"{n_bytes / (1024 * 1024):.1f} MiB"

def print_benchmark_table(results: List[dict]):
    """
    Print results in a Markdown table using tabulate.
    
    results is a list of dicts with keys:
    'Variable', 'Decompressed', 'Compressed', 'Ratio', 'FS Error', 'Write Speed', 'Read Speed'
    """
    try:
        from tabulate import tabulate
    except ImportError:
        print("tabulate not installed. Printing raw results.")
        for res in results:
            print(res)
        return

    headers = list(results[0].keys())
    rows = [list(res.values()) for res in results]
    print(tabulate(rows, headers=headers, tablefmt="github"))
