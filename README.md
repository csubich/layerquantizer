# LayerQuantizer: put your ERA5 on a diet

LayerQuantizer is an offshoot of a [research project](https://journals.ametsoc.org/view/journals/aies/4/3/AIES-D-24-0101.1.xml) and 
[AI model training codebase](https://github.com/csubich/graphcast), targeting the data side 
of training AI models over gridded weather data sets.

In particular, weather data is big.  The highest-resolution dataset in common use today is 
the ERA5 reanalysis provided by the [WeatherBench 2 project](https://weatherbench2.readthedocs.io/en/latest/data-guide.html), 
and the 3D atmospheric variables in that dataset are set on a ¼° latitude/longitude grid with
37 vertical levels.  This means that each 3D variable is about 150 MiB uncompressed, and
an AI model training over this dataset needs an effective input bandwidth of O(1GiB/sec).

Operational weather centres routinely face this problem, and LayerQuantizer is an adaptation
of [ECCC's](https://www.canada.ca/en/environment-climate-change.html) compresison algorithm,
applied to Zarr-backed datasets.

## Theory of operation

LayerQuantizer takes a two or three-dimensional (level, latitude, longitude) field and proceeds
as follows:

1. On a per-layer basis, compute the maximum and minimum of the field.
2. Rescale the field's elements by this minimum and maximum and quanitize them to a specified
number of bits (default 16).
3. Apply [Lorenzo encoding](https://www.osti.gov/servlets/purl/15004622) to this now-integer field,
where each element is written as $δ[i,j] ← x[i,j] - x[i-1,j] - x[i,j-1] + x[i-1,j-1]$, or the
residual of a linear prediction based on the lexically-prior neighbours in each dimension.
4. Encode this $δ$ representation in [base negative two](https://en.wikipedia.org/wiki/Negative_base),
ensuring that small values receive a binary representation with many leading zeroes.
5. Pass this encoded field on to a standard Blosc encoder (default zstd, after some mild experimentation)
for entropy coding.

Layer-based quantization is necessary for meteorological values because so many fields have a strong
dependence on height.

## Installation and use

This encoder can be installed via `pip install git+https://github.com/csubich/layerquantizer.git`.
Required dependencies are numcodecs, numba, numpy, and zarr(>=3).  The Zarr dependency implies a baseline
Python version of 3.11 or later, and it works well when combined with xarray.

To use the codec to read Zarr stores that have already been compressed with this codec, _no action is
required_ after installation.  The pyproject.toml file installs the necessary hooks to register the
codec with both numcodecs (Zarr-2) and Zarr 3 directly.  The string representation of the Zarr-2 
(numcodecs) codec is `layerquantizer0.3b`, and the Zarr-3 codec is `layerquantizer0.3`.

To use the codec to _write_ Zarr stores, the approach differs by Zarr version:

### Zarr 2 (numcodecs)

When using the codec to write to a Zarr-2 store, specify it as a compressor for the Zarr array.  The codec
can be specified by name, or a `layerquantizer.LayerQuantizer` object can be instantiated to provide more
control over quantization and compression options.

### Zarr 3

When using the codec to write to a Zarr-3 store, specify it as a _serializer_, not as a compressor.  Zarr-3
separates the roles of a serializer (turning an array into bytes) and a compressor (turning bytes into fewer
bytes), and this project is more at home as a serializer.  To use it, instantiate a 
`layerquantizer.LayerQuantizerCodec` object for the serializer, and pass a suitable compressor from `zarr.codecs`
for compression.  A blosc-based codec is recommended because its built-in byte-shuffle support does very nice
things for the compressibility of quantized data.  

### Opptions

In both cases, LayerQuantizer offers some customizability when instantiated as a compressing or serializing 
object; please see the docstrings for the repsective classes for full detail.  The default options 
correspond to 16-bit quantization and (Zarr-2) blosc-backed zstd level 5 compression with byte shuffling.

If the data is likely to be re-encoded (e.g. liable to be re-chunked), then consider using the `pow2_range`
option.  This expands the layerwise range to the next largest power of 2 when encoding.  This can cost up to
one bit of accuracy (e.g. a field that spans [0,1] inclusive must be encoded as if it spans the [0,2) 
half-open interval), but in return it means that the quantization levels are stable under re-encoding.  Encoding
a subset of the already-quantized grid will result in either the same quantization levels (if the subset range
is close to the original range) or a superset of the same levels (if the subset range is smaller than half
of the expanded-to-pow2 range).  Use of this option is transparent when reading.

### Examples

For examples of the correct approach for both Zarr-2 and Zarr-3 (via xarray), see the [demonstration 
notebook](examples/colab_demo.ipynb), which is suitable for running on Google Colab and performs some basic
benchmarking with ERA5 data downloaded on-demand from WeatherBench.

### Threading (or lack thereof)

The encoding and decoding are accomplished in Python, using numba to compile the heaviest helper functions.
Both stages are generally single-threaded, with the intention to support higher-level parallelization
(such as dask) to read or write many variables/time levels simultaneously in operational use.

## Results

The [demonstration workbook](examples/colab_demo.ipynb) executes on Google colab and compresses
the three-dimensional variables in the ERA5-hourly dataset for a randomly selected set of dates. 

### Compression ratios

LayerQuantizer supports flexible downstream compression types.  For the Zarr-2 LayerQuantizer, the
compression engine and compression level can be passed to `LayerQuantizer`, which in turn is passed-
through to Blosc.  for Zarr-3, compression is handled separately by Zarr.  For a selection of these
options, the achieved compression ratios with 16-bit layer quantization (compared to default settings
and compression without LayerQuantizer) are:

#### Zarr 2: Compression Ratio (Higher is Better)
| Variable            |   Default (LZ4) |   LQ (16, LZ4) |   Zstd 5 |   LQ (16, Zstd 1) |   LQ (16, Zstd 5) |   LQ (16, Zstd 9) |
|---------------------|-----------------|----------------|----------|-------------------|-------------------|-------------------|
| geopotential        |            1.90 |           3.81 |     2.22 |              5.46 |              5.51 |              5.70 |
| potential\_vorticity |            1.40 |           2.81 |     1.62 |              3.62 |              3.72 |              3.95 |
| specific\_humidity   |            1.47 |           2.85 |     1.67 |              3.53 |              3.62 |              3.78 |
| temperature         |            1.80 |           3.16 |     1.98 |              3.74 |              3.85 |              4.07 |
| u\_component\_of\_wind |            1.30 |           3.12 |     1.42 |              3.67 |              3.78 |              4.03 |
| v\_component\_of\_wind |            1.26 |           3.02 |     1.35 |              3.56 |              3.68 |              3.92 |
| vertical\_velocity   |            1.22 |           2.60 |     1.38 |              3.11 |              3.24 |              3.46 |

#### Zarr 3: Compression Ratio (Higher is Better)
| Variable            |   Default (Zstd 0) |   Zstd 5 |   Blosc+Zstd 5 |   LQ (16) + Blosc + Zstd 1 |   LQ (16) + Blosc + Zstd 5 |
|---------------------|--------------------|----------|----------------|----------------------------|----------------------------|
| geopotential        |               1.82 |     1.87 |           2.22 |                       5.46 |                       5.51 |
| potential\_vorticity |               1.95 |     2.04 |           1.62 |                       3.62 |                       3.72 |
| specific\_humidity   |               1.63 |     1.70 |           1.67 |                       3.53 |                       3.62 |
| temperature         |               1.70 |     1.74 |           1.98 |                       3.74 |                       3.85 |
| u\_component\_of\_wind |               1.37 |     1.43 |           1.42 |                       3.67 |                       3.78 |
| v\_component\_of\_wind |               1.35 |     1.42 |           1.35 |                       3.56 |                       3.68 |
| vertical\_velocity   |               1.61 |     1.68 |           1.38 |                       3.10 |                       3.24 |

In each case, the "full scale error", or pointwise error divided by the layerwise range, is less than $0.5 \cdot 2^{-16} \approx 7.6 \cdot 10^{-6}$

## Limitations and warnings

This code serves adequately to train Graphcast-type models, but it's very much an alpha version without
widespread testing.

### Research code

This code was primarily developed to support ongoing research projects, and it should be used in that
spirit.  In particular, *please don't use this for permanent archival*.  The author has not encountered
any data-loss bugs, but the further one strays from compresisng data that looks like an Zarr-backed
xarray Dataset of ERA5-like fields the more one goes into uncharted territory.

### Semantic versioning

This project has adopted semantic versioning based on stability of the on-disk format:

* Incrementing the _major_ version (currently 0) may signal a breaking change, with the library no longer able
to read files created with older versions.
* Incrementing the _minor_ version (currently 3) signals backwards-compatible changes to the disk representation.
Generally this would come through adding new stringified codec names and including registration of compatible
read/write versions of the codec.  That is, a hypothetical version 0.4 will be able to read and write version
0.3 files, but it will probably also be able to read and write version 0.4 files that can not be read by version
0.3.  Files that were validly written by 0.3.x will retain the same values when decoded by version 0.4.x, up to
roundoff-level changes.
* Incrementing the _patch_ version (currently 2) signals bugfixes, improvements, or new features that do not affect
existing on-disk representations of validly-written files.  This version increment might fix file corruption issues,
however, where arrays that would have resulted in an erroneous on-disk representation are now written correctly.  Such
changes should generally be _forwards compatible_, where an array written with 0.3.b can be read by 0.3.a (b\>a).

### Gridded, float32 data only

This codec makes strong assumptions that the underlying array is gridded with two or more dimensions, with
the numpy ordering of (…,x,y) [typically latitude/longitude].  The two spatial dimensions are equivalent
as far as the compressor is concerned, and the layer-based quantization is broadcast to all higher-order
dimensions without interaction.  No sanity checking is performed to make sure that the input data matches
this format, and this might cause precision loss if the natural layer structure for quantization does not
meet the assumed dimension ordering.

The code also currently only supports float32 data, and supplying any other data type is likely to result
in an exception.  Extending it to integer data might also be interesting, since there the quantization is
irrelevant and it would act as a lossless compressor.

### Non-finite support

This code supports NaN values in the uncompressed stream (giving them an out-of-range integer representation
after quantization), but it treats all NaN values as equivalent and thus loses any information contained in
NaN-tagging.  This support exists primarily to encode missing or null values in the representation, since
Zarr implicitly nan/null/fill-pads arrays if their logical size does not divide evenly into chunks.

This code does _not_ currently support positive and negative infinity values, although it could be extended
to do so relatively easily.  Denormal data is also likely to be flushed to zero at the quantization stage, since
the numba kernels are compiled with fast-math options.

If encoding a dataset that uses a sigil value for "null or not present," consider replacing that value with
a NaN in a preprocessing step.  If the sigil value is numerical but out of the typical range, the quantization
step will deicde that it needs to represent all of the nonexistent values between the sigil and the valid data,
losing considerable precision.

### Carefully select compressed variables

Layer-based quantization is appropriate for most 3D atmospheric fields, but please carefully consider the
impacts for other variables.  For example, variables like land surface type that are really categorical
variables inside a float32 wrapper shouldn't be quantized (what's a land type of 1.0004 mean?), and varibles
like precipitation that have  a wide dynamic range and something closer to a log or power-law distribution
might see unexpected relative errors, even if the absolute error caused by quantization is managed.

When encoding a 'proper' training dataset, it would be reasonable to use LayerQuantizer only for the 3D
atmospheric variables, leaving the 2D ones for the default lossless compression.  The 3D variables are
responsible for the lion's share of the dataset size.

### Speed

This codec is primarily implemented in Python, with compilation of the quantization and encoding kernels via
Numba.  This imples a small performance penalty relative to an optimized implementation, but as of version 0.3.2
kernel fusion has sped thigs up significantly such that the bottleneck is generally the back-end compression
(with zstd giving good compression ratios at a write-speed penalty). 

The demonstration colab notebook includes a performance benchmark for various configurations using an in-memory
Zarr store to eliminate disk bandwidth as a bottleneck.  

Performance was measured based on the in-memory (uncompressed) data size, reflecting the developer intuition of
"how long will it take to read/write an array of a given size?"  In testing on colab and aggregating over the
above 3D ERA5 variables, these performance results were:

#### Zarr 2

| Config          |   Write (MiB/s) |   Read (MiB/s) |
|-----------------|-----------------|----------------|
| Default (LZ4)   |           396.3 |          442.7 |
| LQ (16, LZ4)    |           373.2 |          360.2 |
| Zstd 5          |            56.6 |          412.2 |
| LQ (16, Zstd 1) |           265.1 |          326.2 |
| LQ (16, Zstd 5) |            68.0 |          307.4 |
| LQ (16, Zstd 9) |             5.0 |          295.6 |

#### Zarr 3a

| Config                   |   Write (MiB/s) |   Read (MiB/s) |
|--------------------------|-----------------|----------------|
| Default (Zstd 0)         |            60.0 |          297.2 |
| Zstd 5                   |            42.3 |          283.7 |
| Blosc+Zstd 5             |            55.6 |          407.2 |
| LQ (16) + Blosc + Zstd 1 |           207.9 |          255.1 |
| LQ (16) + Blosc + Zstd 5 |            63.6 |          283.3 |

(Note that Zarr switched from LZ4 as its default compressor to Zstd-level-0 with Zarr-3.  Blosc, however, reserves
compression level 0 for an uncompressed 'store' operation and does not pass it to a zstd backend.)

## License and citation

This code is available under the Apache 2.0 license, and as a Government of Canada work product it is copyright
the Crown in right of Canada.

If you use this code in your research, the author would appreciate a citation to either this github repository or
to [C. Subich, “Efficient Fine-Tuning of 37-Level GraphCast with the Canadian Global Deterministic Analysis,” July 2025, 
doi: 10.1175/AIES-D-24-0101.1.](https://journals.ametsoc.org/view/journals/aies/4/3/AIES-D-24-0101.1.xml), the research
project for which the code was created.  This section will be updated if any article or technical report that more
properly describes this codec is published.
