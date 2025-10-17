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
Required dependencies are numcodecs, numba, and numpy, but the encoder is most useful when combined
with Zarr-python and xarray.

To use the codec, `import layerquantizer`.  The import code registers the codec with `numcodecs`, and
that suffices when reading an on-disk Zarr that has one or more fields so-encoded.

When using codec to write data to disk, specify it as a compressor for a Zarr array (version 2).  Once
registered the codec can be specified by name (`layerquantizer0.3b`), or a `layerquantizer.LayerQuantizer`
object can be instantiated to provide more control over the process.

The encoding and decoding are accomplished in Python, using numba to compile the heaviest helper functions.
Both stages are generally single-threaded, with the intention to support higher-level parallelization
(such as dask) to read or write many variables/time levels simultaneously in operational use.

## Results

The [included demonstration workbook](layerquantizer_demo.ipynb) executes on Google colab and compresses
the three-dimensional variables in the ERA5-hourly dataset for a randomly selected set of dates.  The
per-variable results are as follows:

Variable | Memory Size | Default ratio | LQ ratio | FS err
:-:|:-:|:-:|:-:|:-:
geopotential | $1465$ MiB | $1.9$ ($766$ MiB) |$5.5$ ($267$ MiB) | $7.81 \\cdot 10^{-6}$
potential_vorticity | $1465$ MiB | $1.4$ ($1049$ MiB) |$3.7$ ($396$ MiB) | $7.66 \\cdot 10^{-6}$
specific_humidity | $1465$ MiB | $1.5$ ($991$ MiB) |$3.6$ ($407$ MiB) | $7.71 \\cdot 10^{-6}$
temperature | $1465$ MiB | $1.8$ ($811$ MiB) |$3.8$ ($383$ MiB) | $7.78 \\cdot 10^{-6}$
u_component_of_wind | $1465$ MiB | $1.3$ ($1121$ MiB) |$3.8$ ($388$ MiB) | $7.69 \\cdot 10^{-6}$
v_component_of_wind | $1465$ MiB | $1.3$ ($1160$ MiB) |$3.7$ ($401$ MiB) | $7.69 \\cdot 10^{-6}$
vertical_velocity | $1465$ MiB | $1.2$ ($1203$ MiB) |$3.2$ ($455$ MiB) | $7.68 \\cdot 10^{-6}$

where "Memory Size" is the total in-memory size of the uncompressed float32 data, "Default ratio" is
the baseline compression ratio achieved by `xarray.Dataset.to_zarr(…,zarr_format=2)` with default
options, "LQ ratio" is the compression ratio achieved with LayerQuantizer, and "FS err" is the 'full
scale error', or maximum error relative to the layerwise (max-min) variation.

## Limitations and warnings

This code serves adequately to train Graphcast-type models, but it's very much an alpha version without
widespread testing.

### Research code

This code was primarily developed to support ongoing research projects, and it should be used in that
spirit.  In particular, *please don't use this for permanent archival*.  The author has not encountered
any data-loss bugs, but the further one strays from compresisng data that looks like an Zarr-backed
xarray Dataset of ERA5-like fields the more one goes into uncharted territory.

The intent is for any future versions of this module to support existing data in a backwards-compatible
manner, by incrementing the version number of the codec upon registry with numcodecs and supplying a
default implementation for deprecated (older) versions.  However, this cannot be guaranteed.

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
to do so relatively easily.

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

This codec is primarily implemented in Python.  Even with the speedups from numba compilation, the performance
is limited compared to a more optimal, purely-compiled version.  In the demonstration colab notebook, the
default and LayerQuantizer data rates (to/from a `zarr.storage.MemoryStore`) were:

| Type | Default | LayerQuantizer
:-:|:-:|:-:|
Read | 562 MiB/sec | 269 MiB/sec
Write | 679 MiB/sec | 59.6 MiB/sec

… when measured based on the in-memory (fully uncompressed) data size.  The slowdown is noticeable over default
compression settings, but it can be mitigated with sufficient outer parallelism – particularly for areas like
model training where a dataset is read far more often than it is written.  

If you are in the fortunate position of having unlimited storage space without an I/O bandwidth bottleneck,
LayerQuantizer may not be useful for you.

Both reading and writing can likely be optimized further.  The current code makes separate passes over the field
for the quantization, Lorenzo encoding, and negabinary representation steps, and they can be combined.

## License and citation

This code is available under the Apache 2.0 license, and as a Government of Canada work product it is copyright
the Crown in right of Canada.

If you use this code in your research, the author would appreciate a citation to either this github repository or
to [C. Subich, “Efficient Fine-Tuning of 37-Level GraphCast with the Canadian Global Deterministic Analysis,” July 2025, 
doi: 10.1175/AIES-D-24-0101.1.](https://journals.ametsoc.org/view/journals/aies/4/3/AIES-D-24-0101.1.xml), the research
project for which the code was created.  This section will be updated if any article or technical report that more
properly describes this codec is published.
