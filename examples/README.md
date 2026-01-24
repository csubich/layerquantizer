# LayerQuantizer Examples and Benchmarks

This directory contains examples and benchmark scripts for demonstrating the performance and usage of `LayerQuantizer` with both Zarr 2 and Zarr 3.

## Overview

The benchmarks use the [WeatherBench 2 ERA5](https://weatherbench2.readthedocs.io/en/latest/index.html) dataset, which is a common benchmark for weather and climate modeling. The scripts compare standard compression (Blosc) with `LayerQuantizer` enhanced compression.

### Scripts

- `benchmark_zarr2.py`: Benchmarks `LayerQuantizer` using the Zarr 2 (Numcodecs) API.
- `benchmark_zarr3.py`: Benchmarks `LayerQuantizer` using the Zarr 3 (Codec API) with "chained" codecs.
- `utils.py`: Shared utilities for data loading, caching, and result formatting.

## Usage

You can run the benchmarks using `uv run`.

### Running Zarr 2 Benchmark

```bash
cd layerquantizer
uv run python examples/benchmark_zarr2.py --ntime 5 --resolution coarse
```

### Running Zarr 3 Benchmark

```bash
cd layerquantizer
uv run python examples/benchmark_zarr3.py --ntime 5 --resolution coarse
```

### Command-Line Arguments

- `--ntime`: Number of random time slices to sample (default: 10).
- `--resolution`: Dataset resolution, either `coarse` (5.625°) or `fine` (0.25°) (default: `coarse`).
- `--mode`: Benchmark mode, either `simple` (default settings) or `sweep` (tests multiple bit rates and compression levels) (default: `simple`).
- `--clear-cache`: Clear the local data cache before running the benchmark.

## Data Caching

The scripts automatically download and cache WeatherBench data to `data/cache/` in the project root. This speeds up subsequent runs. If you encounter issues with corrupted data or want to free up space, use the `--clear-cache` flag.

## Metrics Explained

- **Decompressed (MiB)**: The size of the raw data in memory.
- **Compressed (MiB)**: The size of the data as stored in the Zarr `MemoryStore`.
- **Ratio**: The compression ratio (Decompressed / Compressed).
- **FS Error**: The Full-Scale Error, defined as $\max\left(\frac{|reconstructed - original|}{\max(layer) - \min(layer)}\right)$.
- **Write (MiB/s)**: Throughput during the writing process.
- **Read (MiB/s)**: Throughput during the reading process.
