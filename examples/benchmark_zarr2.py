import argparse
import datetime
import asyncio
import xarray as xr
import zarr
import numpy as np
import sys
import os

# Add parent directory to sys.path to allow importing from examples.utils if run from within examples/
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import get_wb_era5, sample_times, get_store_size, VARIABLES_3D, print_benchmark_table, print_summary_tables, clear_cache

from layerquantizer import LayerQuantizer
from numcodecs import Blosc

async def run_benchmark(args):
    if args.clear_cache:
        clear_cache()
    print(f"Loading WeatherBench ERA5 ({args.resolution}) dataset...")
    ds_full = get_wb_era5(resolution=args.resolution)
    
    print(f"Sampling {args.ntime} time slices...")
    ds = sample_times(ds_full, n_time=args.ntime)
    
    results = []
    
    # Define configurations to test
    configs = []
    if args.mode == "simple":
        configs = [
            ("Blosc(zstd, clevel=5)", Blosc(cname='zstd', clevel=5, shuffle=Blosc.SHUFFLE)),
            ("LQ (16-bit)", LayerQuantizer(nbits=16)),
        ]
    elif args.mode == "sweep":
        # Test Blosc(zstd) alone with various clevels
        for clevel in [1, 5, 9]:
            configs.append((f"Blosc(zstd, clevel={clevel})", 
                            Blosc(cname='zstd', clevel=clevel, shuffle=Blosc.SHUFFLE)))
        # Test LayerQuantizer + Blosc(zstd) with various nbits and clevels
        for nbits in [8, 12, 16]:
            for clevel in [1, 5, 9]:
                configs.append((f"LQ (nbits={nbits}, clevel={clevel})", 
                                LayerQuantizer(nbits=nbits, blosc_clevel=clevel)))

    for var_name in VARIABLES_3D:
        if var_name not in ds.data_vars:
            print(f"Variable {var_name} not found in dataset. Skipping.")
            continue
            
        print(f"\nBenchmarking variable: {var_name}")
        # Process one variable at a time to save memory
        dv = ds[[var_name]].compute()
        raw_bytes = dv.nbytes
        
        for label, compressor in configs:
            print(f"  Testing {label}...")
            store = zarr.storage.MemoryStore()
            
            encoding = {}
            if compressor:
                encoding = {var_name: {"compressor": compressor}}
            
            # Write benchmark
            start_write = datetime.datetime.now()
            dv.to_zarr(store, compute=True, encoding=encoding, zarr_format=2)
            write_time = (datetime.datetime.now() - start_write).total_seconds()
            
            # Read benchmark
            start_read = datetime.datetime.now()
            reload = xr.open_zarr(store).compute()
            read_time = (datetime.datetime.now() - start_read).total_seconds()
            
            # Metrics
            comp_bytes = await get_store_size(store)
            ratio = raw_bytes / comp_bytes if comp_bytes > 0 else 0
            
            # Error metric (Full-Scale Error)
            full_scale_diff = dv[var_name].max() - dv[var_name].min()
            abs_diff = np.abs(reload[var_name] - dv[var_name]).max()
            fs_err = float(abs_diff / full_scale_diff) if full_scale_diff > 0 else 0
                
            write_speed = raw_bytes / (1024 * 1024) / write_time
            read_speed = raw_bytes / (1024 * 1024) / read_time
            
            results.append({
                "Variable": var_name,
                "Config": label,
                "Decompressed (MiB)": raw_bytes / (1024*1024),
                "Compressed (MiB)": comp_bytes / (1024*1024),
                "Ratio": ratio,
                "FS Error": fs_err,
                "Write (MiB/s)": write_speed,
                "Read (MiB/s)": read_speed
            })

    print("\nBenchmark Results (Zarr 2):")
    print_benchmark_table(results)
    print_summary_tables(results)
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark LayerQuantizer with Zarr 2")
    parser.add_argument("--ntime", type=int, default=10, help="Number of time slices to sample")
    parser.add_argument("--resolution", choices=["coarse", "fine"], default="coarse", help="Dataset resolution")
    parser.add_argument("--mode", choices=["simple", "sweep"], default="simple", help="Benchmark mode")
    parser.add_argument("--clear-cache", action="store_true", help="Clear the local data cache before running")
    
    args = parser.parse_args()
    asyncio.run(run_benchmark(args))
