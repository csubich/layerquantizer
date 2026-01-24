import argparse
import asyncio
import sys
import os

# Add parent directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import print_benchmark_table, print_summary_tables, clear_cache
from benchmark_zarr2 import run_benchmark as run_zarr2
from benchmark_zarr3 import run_benchmark as run_zarr3

async def main():
    parser = argparse.ArgumentParser(description="Run all LayerQuantizer benchmarks (Zarr 2 and Zarr 3)")
    parser.add_argument("--ntime", type=int, default=10, help="Number of time slices to sample")
    parser.add_argument("--resolution", choices=["coarse", "fine"], default="coarse", help="Dataset resolution")
    parser.add_argument("--mode", choices=["simple", "sweep"], default="simple", help="Benchmark mode")
    parser.add_argument("--clear-cache", action="store_true", help="Clear the local data cache before running")
    
    args = parser.parse_args()
    
    if args.clear_cache:
        clear_cache()
        # Ensure we don't clear it again in the sub-benchmarks
        args.clear_cache = False

    print("=== Running Zarr 2 Benchmarks ===")
    results_z2 = await run_zarr2(args)
    for r in results_z2:
        r["Zarr"] = "2"

    print("\n=== Running Zarr 3 Benchmarks ===")
    results_z3 = await run_zarr3(args)
    for r in results_z3:
        r["Zarr"] = "3"

    print("\n=== Consolidated Benchmark Results ===")
    all_results = results_z2 + results_z3
    
    # Reorder columns to put Zarr version near the front
    final_results = []
    for r in all_results:
        ordered_r = {
            "Variable": r["Variable"],
            "Zarr": r["Zarr"],
            "Config": r["Config"],
            "Decompressed (MiB)": r["Decompressed (MiB)"],
            "Compressed (MiB)": r["Compressed (MiB)"],
            "Ratio": r["Ratio"],
            "FS Error": r["FS Error"],
            "Write (MiB/s)": r["Write (MiB/s)"],
            "Read (MiB/s)": r["Read (MiB/s)"]
        }
        final_results.append(ordered_r)
        
    print("\nDetailed Results:")
    print_benchmark_table(final_results)
    
    print("\nSummary Tables:")
    print_summary_tables(final_results)

if __name__ == "__main__":
    asyncio.run(main())
