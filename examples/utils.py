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
    'Variable', 'Decompressed (MiB)', 'Compressed (MiB)', 'Ratio', 'FS Error', 'Write (MiB/s)', 'Read (MiB/s)'
    Values can be strings or numbers.
    """
    try:
        from tabulate import tabulate
    except ImportError:
        print("tabulate not installed. Printing raw results.")
        for res in results:
            print(res)
        return

    if not results:
        return

    # Create a copy for formatting
    formatted_results = []
    for r in results:
        fr = {}
        for k, v in r.items():
            if isinstance(v, float):
                if "Ratio" in k:
                    fr[k] = f"{v:.2f}x"
                elif "Error" in k:
                    fr[k] = f"{v:.2e}"
                elif "MiB" in k or "MiB/s" in k:
                    fr[k] = f"{v:.1f}"
                else:
                    fr[k] = f"{v:.2f}"
            else:
                fr[k] = v
        formatted_results.append(fr)

    headers = list(formatted_results[0].keys())
    rows = [list(res.values()) for res in formatted_results]
    print(tabulate(rows, headers=headers, tablefmt="github"))

def print_summary_tables(results: List[dict]):
    """
    Print pivoted summary tables as requested in Task 12.
    """
    try:
        from tabulate import tabulate
    except ImportError:
        return

    if not results:
        return

    # Ensure all results have a 'FullConfig'
    for r in results:
        if "FullConfig" not in r:
            z = f"Z{r['Zarr']} " if "Zarr" in r else ""
            r["FullConfig"] = f"{z}{r['Config']}".strip()

    variables = []
    for r in results:
        if r["Variable"] not in variables:
            variables.append(r["Variable"])
    
    configs = []
    for r in results:
        if r["FullConfig"] not in configs:
            configs.append(r["FullConfig"])

    # 1. Compression Ratio Table
    ratio_headers = ["Variable"] + configs
    ratio_rows = []
    for var in variables:
        row = [var]
        for cfg in configs:
            match = next((r for r in results if r["Variable"] == var and r["FullConfig"] == cfg), None)
            if match:
                val = match["Ratio"]
                row.append(f"{val:.2f}x" if isinstance(val, (int, float)) else val)
            else:
                row.append("N/A")
        ratio_rows.append(row)
    
    print("\n### Compression Ratio (Higher is Better)")
    print(tabulate(ratio_rows, headers=ratio_headers, tablefmt="github"))

    # 2. Performance Table (Speed)
    speed_headers = ["Metric"] + configs
    write_row = ["Write (MiB/s)"]
    read_row = ["Read (MiB/s)"]
    
    for cfg in configs:
        cfg_results = [r for r in results if r["FullConfig"] == cfg]
        if cfg_results:
            write_speeds = [r["Write (MiB/s)"] for r in cfg_results if isinstance(r["Write (MiB/s)"], (int, float))]
            read_speeds = [r["Read (MiB/s)"] for r in cfg_results if isinstance(r["Read (MiB/s)"], (int, float))]
            write_row.append(f"{np.mean(write_speeds):.1f}" if write_speeds else "N/A")
            read_row.append(f"{np.mean(read_speeds):.1f}" if read_speeds else "N/A")
        else:
            write_row.append("N/A")
            read_row.append("N/A")

    print("\n### Performance (Average MiB/s)")
    print(tabulate([write_row, read_row], headers=speed_headers, tablefmt="github"))

    # 3. Full-Scale Error Table
    error_headers = ["Variable"] + configs
    error_rows = []
    for var in variables:
        row = [var]
        for cfg in configs:
            match = next((r for r in results if r["Variable"] == var and r["FullConfig"] == cfg), None)
            if match:
                val = match["FS Error"]
                row.append(f"{val:.2e}" if isinstance(val, (int, float)) else val)
            else:
                row.append("N/A")
        error_rows.append(row)
    
    print("\n### Full-Scale Error (Lower is Better)")
    print(tabulate(error_rows, headers=error_headers, tablefmt="github"))
