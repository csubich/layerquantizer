from __future__ import annotations
import numcodecs
import warnings
import numpy as np
from typing import Any

# Define helper functions for quantization and dequantization
try:
    # If numba is available, use it to create compiled versions of the functions.  The best numba function
    # has a C/Fortran-like structure with loops, so it performs poorly in raw Python, but the best vector
    # function does not gain much from numba
    import numba

    numba.config.THREADING_LAYER = "threadsafe"

    # Numba note: fastmath=True causes LLVM to assume that nans don't exist, so this will require some
    # care when quantizing actual nan values (missing/out-of-bounds data)
    @numba.jit(
        [
            numba.void(
                numba.types.Array(numba.types.float32, 3, "C", readonly=True),
                numba.float32[:],
                numba.float32[:],
            )
        ],
        nopython=True,
        nogil=True,
    )
    def get_plane_extrema(
        buf: np.ndarray, plane_min: np.ndarray, plane_max: np.ndarray
    ) -> None:
        """Calculate per-plane minima and maxima in a single pass."""
        Nplanes, Ni, Nj = buf.shape
        for kk in numba.prange(Nplanes):
            p_min = np.float32(np.inf)
            p_max = np.float32(-np.inf)
            any_valid = False
            for jj in range(Ni):
                for ii in range(Nj):
                    v = buf[kk, jj, ii]
                    # Check for NaN.  np.isnan is safe here since we don't use fastmath
                    if not np.isnan(v):
                        if v < p_min:
                            p_min = v
                        if v > p_max:
                            p_max = v
                        any_valid = True
            if not any_valid:
                plane_min[kk] = 0
                plane_max[kk] = 0
            else:
                plane_min[kk] = p_min
                plane_max[kk] = p_max

    @numba.jit(
        [
            numba.void(
                numba.types.Array(numba.types.float32, 3, "C", readonly=True),
                numba.int64,
                numba.float32[:],
                numba.float32[:],
                numba.bool_,
                numba.int32[:, :, :],
            )
        ],
        nopython=True,
        fastmath=True,
        nogil=True,
    )
    def quantize_kernel(
        buf: np.ndarray,
        nbits: int,
        plane_min: np.ndarray,
        plane_max: np.ndarray,
        do_lorenzo: bool,
        out: np.ndarray,
    ) -> None:
        """Encode buffer through linear quantization and optionally Lorenzo prediction,
        returning the base-negative-two encoded results in a single pass."""

        Nplanes = buf.shape[0]
        Ni = buf.shape[1]
        Nj = buf.shape[2]

        # Maximum quantized level (0 -- MAX_LEVEL-1 inclusive, giving 2**nbits values)
        MAX_LEVEL = numba.int32(2**nbits - 1)
        # Sigil value for NaNs
        NAN_SIGIL = numba.int32(MAX_LEVEL + 1)
        MAX_LEVELf = np.float32(MAX_LEVEL)

        Schroeppel2 = np.uint32(0xAAAAAAAA)

        for kk in numba.prange(Nplanes):
            plane_delta = plane_max[kk] - plane_min[kk]
            if plane_delta <= 0:
                plane_delta = 1
            plane_scale = MAX_LEVELf / plane_delta

            if do_lorenzo:
                # Buffer for the previous row of raw quantized values
                prev_q_row = np.empty(Nj, dtype=np.int32)

                for jj in range(Ni):
                    q_left = np.int32(0)
                    q_upleft = np.int32(0)
                    for ii in range(Nj):
                        # 1. Quantize
                        if (buf.view(np.int32)[kk, jj, ii] & 0x7F80_0000) == 0x7F80_0000:
                            q_curr = NAN_SIGIL
                        else:
                            q_curr = np.int32(
                                np.rint(plane_scale * (buf[kk, jj, ii] - plane_min[kk]))
                            )

                        # 2. Lorenzo
                        if jj == 0:
                            if ii == 0:
                                l_curr = q_curr
                            else:
                                l_curr = q_curr - q_left
                        else:
                            if ii == 0:
                                l_curr = q_curr - prev_q_row[ii]
                            else:
                                l_curr = q_curr - prev_q_row[ii] - q_left + q_upleft

                        # Update buffers for next iteration
                        q_upleft = prev_q_row[ii]
                        prev_q_row[ii] = q_curr
                        q_left = q_curr

                        # 3. Negabinary
                        bu32 = np.uint32(l_curr)
                        out[kk, jj, ii] = np.int32((bu32 + Schroeppel2) ^ Schroeppel2)
            else:
                for jj in range(Ni):
                    for ii in range(Nj):
                        if (buf.view(np.int32)[kk, jj, ii] & 0x7F80_0000) == 0x7F80_0000:
                            out[kk, jj, ii] = NAN_SIGIL
                        else:
                            out[kk, jj, ii] = np.int32(
                                np.rint(plane_scale * (buf[kk, jj, ii] - plane_min[kk]))
                            )

    def quantizer(
        buf: np.ndarray, nbits: int, plane_min: np.ndarray, plane_max: np.ndarray
    ) -> np.ndarray:
        out = np.empty(buf.shape, dtype=np.int32)
        quantize_kernel(buf, nbits, plane_min, plane_max, False, out)
        return out

    @numba.jit(
        [
            numba.void(
                numba.types.Array(numba.types.int32, 3, "C", readonly=True),
                numba.float32[:, :, :],
                numba.types.Array(numba.types.float32, 1, "C", readonly=True),
                numba.float32[:],
                numba.int64,
                numba.bool_,
            )
        ],
        nopython=True,
        fastmath=True,
        nogil=True,
    )
    def dequantize_kernel(
        int_field: np.ndarray,
        out: np.ndarray,
        plane_min: np.ndarray,
        plane_delta: np.ndarray,
        nbits: int,
        do_lorenzo: bool,
    ) -> None:
        """Decode buffer through base-negative-two decoding, inverse Lorenzo,
        and dequantization in a single pass."""
        Nplanes = int_field.shape[0]
        Ni = int_field.shape[1]
        Nj = int_field.shape[2]

        MAX_LEVEL = numba.int32(2**nbits - 1)
        NAN_SIGIL = numba.int32(MAX_LEVEL + 1)
        inv_max = np.float32(1.0 / MAX_LEVEL)

        Schroeppel2 = np.uint32(0xAAAAAAAA)

        for kk in numba.prange(Nplanes):
            delta = inv_max * plane_delta[kk]
            p_min = plane_min[kk]

            if do_lorenzo:
                prev_q_row = np.empty(Nj, dtype=np.int32)

                for jj in range(Ni):
                    csum = np.int32(0)
                    for ii in range(Nj):
                        # 1. Binanegary
                        bu32 = np.uint32(int_field[kk, jj, ii])
                        l_val = np.int32((bu32 ^ Schroeppel2) - Schroeppel2)

                        # 2. Unlorenzo
                        csum += l_val
                        if jj == 0:
                            q = csum
                        else:
                            q = prev_q_row[ii] + csum

                        prev_q_row[ii] = q

                        # 3. Rescale
                        if q == NAN_SIGIL:
                            out[kk, jj, ii] = np.nan
                        else:
                            out[kk, jj, ii] = p_min + delta * q
            else:
                for jj in range(Ni):
                    for ii in range(Nj):
                        q = int_field[kk, jj, ii]
                        if q == NAN_SIGIL:
                            out[kk, jj, ii] = np.nan
                        else:
                            out[kk, jj, ii] = p_min + delta * q

    def dequantizer(
        buf: np.ndarray, nbits: int, plane_min: np.ndarray, plane_max: np.ndarray
    ) -> np.ndarray:
        out = np.empty(buf.shape, dtype=np.float32)
        dequantize_kernel(buf, out, plane_min, plane_max - plane_min, nbits, False)
        return out
except ImportError:
    # numba isn't available, so define vector functions as a fallback
    def get_plane_extrema(
        buf: np.ndarray, plane_min: np.ndarray, plane_max: np.ndarray
    ) -> None:
        """Calculate per-plane minima and maxima using numpy."""
        with warnings.catch_warnings():
            # Suppress a warning message if an entire plane is nans
            warnings.filterwarnings("ignore", r"All-NaN (slice|axis) encountered")
            plane_min[:] = np.nanmin(buf, axis=(1, 2))
            plane_max[:] = np.nanmax(buf, axis=(1, 2))
        # If an entire plane is nan (can happen with chunking), set the min and max to both be 0
        np.nan_to_num(plane_min, copy=False, nan=0)
        np.nan_to_num(plane_max, copy=False, nan=0)

    def quantize_kernel(
        buf: np.ndarray,
        nbits: int,
        plane_min: np.ndarray,
        plane_max: np.ndarray,
        do_lorenzo: bool,
        out: np.ndarray,
    ) -> None:
        """Encode buffer through quantization and linear prediction using numpy."""

        # Maximum quantized level (0 -- MAX_LEVEL-1 inclusive, giving 2**nbits values)
        MAX_LEVEL = np.int32(2**nbits - 1)
        # Sigil value for NaNs
        NAN_SIGIL = np.int32(MAX_LEVEL + 1)
        MAX_LEVELf = float(MAX_LEVEL)

        plane_delta = np.where((plane_max - plane_min) > 0, plane_max - plane_min, 1)

        # Quantize the array per-plane
        plane_scale = MAX_LEVELf / plane_delta
        quantized_f = np.rint(
            plane_scale[:, None, None] * (buf - plane_min[:, None, None])
        )
        # Mark any NANs by the sigil value
        np.nan_to_num(quantized_f, copy=False, nan=float(NAN_SIGIL))
        quantized_int = quantized_f.astype(np.int32)

        if do_lorenzo:
            quantized_int = negabinary(lorenzo2d(quantized_int))

        out[:] = quantized_int

    quantizer = None

    def dequantize_kernel(
        buf: np.ndarray,
        out: np.ndarray,
        plane_min: np.ndarray,
        plane_delta: np.ndarray,
        nbits: int,
        do_lorenzo: bool,
    ) -> None:
        """Takes quantized integer values and re-scale them to their float32 equivalents"""
        if do_lorenzo:
            buf = unlorenzo2d(binanegary(buf.view(np.uint32)))

        MAX_LEVEL = np.int32(2**nbits - 1)
        NAN_SIGIL = np.int32(MAX_LEVEL + 1)

        delta = (plane_delta / MAX_LEVEL).astype(np.float32)
        out_val = buf * delta[:, None, None] + plane_min[:, None, None]
        out_val[buf == NAN_SIGIL] = np.nan
        out[:] = out_val

    dequantizer = None


@numba.vectorize([numba.uint32(numba.int32)], nopython=True)
def negabinary(binary: Any) -> Any:
    """Encode a signed 32-bit value (or array thereof) into base negative two,
    following https://en.wikipedia.org/wiki/Negative_base#Shortcut_calculation"""
    Schroeppel2 = np.uint32(0xAAAAAAAA)
    bu32 = np.uint32(binary)
    return (bu32 + Schroeppel2) ^ Schroeppel2


@numba.vectorize([numba.int32(numba.uint32)], nopython=True)
def binanegary(negabinary: Any) -> Any:
    """Convert a 32-bit value from base negative two to two's complement (signed) form"""
    Schroeppel2 = np.uint32(0xAAAAAAAA)
    bu32 = negabinary
    return (bu32 ^ Schroeppel2) - Schroeppel2


@numba.jit(
    [
        numba.int32[:, :, :](
            numba.types.Array(numba.types.int32, 3, "A", aligned=True, readonly=True)
        )
    ],
    nopython=True,
    nogil=True,
)
def lorenzo2d(A: np.ndarray) -> np.ndarray:
    """Perform Loernzo encoding (lexical prediction based on S/W/SW values) on a 2D array"""
    out = np.zeros_like(A)
    for k in range(0, A.shape[0]):
        for i in range(1, A.shape[2]):
            out[k, 0, i] = A[k, 0, i] - A[k, 0, i - 1]
        for j in numba.prange(1, A.shape[1]):
            out[k, j, 0] = A[k, j, 0] - A[k, j - 1, 0]
            for i in range(1, A.shape[2]):
                out[k, j, i] = (
                    A[k, j, i] - A[k, j - 1, i] - A[k, j, i - 1] + A[k, j - 1, i - 1]
                )
        out[k, 0, 0] = A[k, 0, 0]
    return out


@numba.jit([numba.int32[:, :, :](numba.int32[:, :, :])], nopython=True, nogil=True)
def unlorenzo2d(A: np.ndarray) -> np.ndarray:
    """Invert Lorenzo encoding on a 2D array"""
    out = np.zeros_like(A)
    Nk = A.shape[0]
    Nj = A.shape[1]
    Ni = A.shape[2]

    for kk in range(Nk):
        # jj=0 case
        out[kk, 0, :] = np.cumsum(A[kk, 0, :])
        for jj in range(1, Nj):
            csum = 0
            for ii in range(Ni):
                csum += A[kk, jj, ii]
                out[kk, jj, ii] = out[kk, jj - 1, ii] + csum

    return out


@numba.jit(
    [
        numba.float32[:, :, :](
            numba.types.Array(numba.types.float32, 3, "C", aligned=True),
            numba.types.Array(numba.types.int32, 3, "C", readonly=True),
            numba.types.Array(numba.types.float32, 1, "C", readonly=True),
            numba.types.Array(numba.types.float32, 1, "C", readonly=True),
            numba.int64,
        )
    ],
    nopython=True,
    nogil=True,
    fastmath=True,
)
def rescale_output(
    outbuf: np.ndarray,
    quantized_field: np.ndarray,
    plane_delta: np.ndarray,
    plane_min: np.ndarray,
    nbits: int,
) -> np.ndarray:
    """Rescale the quantized output back to float32 given the quantized field,
    the per-plane minima and deltas, and the number of bytes in the quantization"""
    MAX_LEVEL = 2**nbits - 1
    NAN_SIGIL = MAX_LEVEL + 1
    inv_max = 1 / MAX_LEVEL

    for kk in range(outbuf.shape[0]):
        for jj in range(outbuf.shape[1]):
            for ii in range(outbuf.shape[2]):
                q = quantized_field[kk, jj, ii]
                if q == NAN_SIGIL:
                    outbuf[kk, jj, ii] = np.nan
                else:
                    outbuf[kk, jj, ii] = plane_min[kk] + inv_max * plane_delta[kk] * q
    # # out[:,:,:] = plane_min[:,None,None] + numba.float32(inv_max * plane_delta[:,None,None] * quantized_field)
    return outbuf


class LayerQuantizer(numcodecs.abc.Codec):
    """LayerQuantizer: dynamic plane-based quantization and linear prediction

    The LayerQuantizer compressor is inspired by the compression of the .fstd files, where a floating-point
    field undergoes a level-based quantization.  Per level (everything but the last two dimensions), the
    encoder records the field minimum and maximum, then it quantizes the field using `nbits` bits.  Nans
    are enoded as (2**nbits + 1).  The resulting integer stream is passed to Blosc for compression.

    By default, the layer minima and maxima are based on their true values, per encoded chunk.  Optionally,
    this can be replaced by fstd-compatible quantization, which expands the quantized range to the next
    largest power of 2.  This prevents data loss when re-encoding these values, whether by encoding an already-
    compressed fstd file or by encoding a subset of a once-encoded layer.

    This quantizer also applies a lossless predictor to the quantized values, to further reduce on-disk storage
    costs beyond what is naively provided by the entropy coder.  The fstd format uses Lorenzo encoding
    (doi:10.1111/1467-8659.00681), which is also used here.  Direct Lorenzo encoding does not result in meaningful
    compression, but combining the Lorenzo-encoded value with a base-negative-two representation does, since it
    packs small-magnitude values (positive or negative) into small-mangnitude unsigned values with long runs of
    binary zero."""

    codec_id = "layerquantizer0.3b"

    def __init__(
        self,
        nbits: int = 16,
        transform: str = "Lorenzo",
        blosc_cname: str = "zstd",
        blosc_clevel: int = 5,
        in_id: str = codec_id,
        pow2_range: bool = False,
    ) -> None:
        super().__init__()
        assert in_id == self.codec_id
        self.nbits = nbits
        self.blosc_cname = blosc_cname  # In testing, zstd was better than lz4
        self.blosc_clevel = blosc_clevel
        self.bloscer = numcodecs.Blosc(
            cname=blosc_cname, clevel=blosc_clevel, shuffle=1
        )
        self.pow2_range = pow2_range
        self.transform = transform

    def encode(self, ibuf: np.ndarray) -> bytes | Any:
        """Encode buffer through layer quantization; each 2D plane of the buffer is quantized
        independently.  This encoder is only valid for float32, so nbits is only meaningful
        for nbits <= 24; any larger value will result in no quantization"""
        assert ibuf.dtype == np.float32
        # Create a view of the input buffer so that shape modifications are non-destructive to the
        # input array
        buf = ibuf.view()

        if self.nbits > 24:
            # Trivial encoding, just apply blosc to the field
            return self.bloscer.encode(buf)

        # Reshape the buffer to (nplanes, ni, nj) format, since the quantization is effectively 3D
        buf.shape = (-1,) + tuple(buf.shape[-2:])
        nplanes = buf.shape[0]

        # Look for the array minimum and maximum by plane
        plane_min = np.empty(nplanes, dtype=np.float32)
        plane_max = np.empty(nplanes, dtype=np.float32)
        get_plane_extrema(buf, plane_min, plane_max)

        # Get the per-plane dynamic range
        plane_delta = plane_max - plane_min

        # Fix up the dynamic range, assigning a minimum range to any plane that had 0 dynamic range
        plane_adjdelta = plane_delta.copy()
        # If min=max=0, then set max=1
        plane_adjdelta[(plane_delta == 0) & (plane_min == 0)] = 1
        # Otherwise, set delta=max(|min|,2*min)
        plane_adjdelta[(plane_delta == 0) & (plane_min != 0)] = np.maximum(
            np.abs(plane_min[(plane_delta == 0) & (plane_min != 0)]),
            2 * plane_min[(plane_delta == 0) & (plane_min != 0)],
        )

        plane_max[plane_delta == 0] = (
            plane_min[plane_delta == 0] + plane_adjdelta[plane_delta == 0]
        ).astype(np.float32)
        plane_delta = plane_adjdelta

        if self.pow2_range:
            # Force range to be a power of 2, for compatibility with fstd quantization.
            # Currently, we know that the field spans [min, max], inclusive of boundaries,
            # but now we want to find new_max such that the field spans [min, new_max)
            # (note _not_ inclusivve of top boundary) and new_amax - min is a power of 2.

            # The net effect is to ensure that (delta)/2**nbits is also a power of 2,
            # so quantization levels are spaced evenly.  Additionally, re-quantizing
            # a subset of this stream will not change the field values, since the
            # quantization delta will either remain the same (if the included range is
            # large enough) or fall by a precise power of 2 (if the included range
            # shrinks sufficiently)

            # However, one side effect is that the field will not use the full range of
            # quantized values.  This effect is most notable for fields like 'cloud fraction'
            # which have a natural range of [0,1] inclusive.  This quantization forces
            # the effective range to [0,2), losing one effective bit.

            # The current fstd code also adjusts the minimum to truncate the law few bits
            # of the mantissa in order to accomplish all of the quantization with integer
            # (fixed-point) math, but this shouldn't be necessary here.  If we're
            # re-encoding an fstd file, we'll already see the truncated minimum.

            # New range value, inclusive of minimum but exclusive of maximum.  The log2
            # would cause a problem if plane_delta were 0, but those cases have been
            # already corrected above
            plane_delta = 2 ** (1 + np.floor(np.log2(plane_delta * (2**self.nbits) / (2**self.nbits - 1))))

            # Use this range to adjust the plane maximum.  plane_max is still
            # notionally inclusive, so it must be adjusted by (2^N-1)/2^N to
            # representt the maximum quantizable value
            plane_max = (
                plane_min + (plane_delta) * (2**self.nbits - 1) / (2**self.nbits)
            ).astype(np.float32)

        # Create the output buffer
        outbuf = np.empty((3 + 2 * nplanes + buf.size), dtype=np.int32)

        # Encode chunk shape
        outbuf[0] = nplanes
        outbuf[1:3] = buf.shape[1:]

        # Encode plane maxima
        outbuf[3 : (3 + nplanes)] = plane_min.view(np.int32)
        outbuf[(3 + nplanes) : (3 + 2 * nplanes)] = plane_max.view(np.int32)

        # Encode the quantized buffer values
        data_view = outbuf[(3 + 2 * nplanes) :].view()
        data_view.shape = buf.shape
        quantize_kernel(
            buf,
            self.nbits,
            plane_min,
            plane_max,
            self.transform == "Lorenzo",
            data_view,
        )

        # Output stream format:
        # [nplanes, nx, ny, mins[nplanes], max[nplanes], bitstream
        return self.bloscer.encode(outbuf)

        # return outbuf

    def decode(self, buf: Any, out: np.ndarray | None = None) -> np.ndarray:
        """Decode the encoded input bytestream"""
        if self.nbits > 24:
            decoded_bytes = self.bloscer.decode(buf)
            res = np.frombuffer(decoded_bytes, dtype=np.float32)
            if out is not None:
                out.ravel()[:] = res.ravel()
                return out
            return res

        intstream = np.frombuffer(self.bloscer.decode(buf), dtype=np.int32)
        # Get chunk size
        nplanes = intstream[0]
        nx = intstream[1]
        ny = intstream[2]

        if out is not None:
            assert out.size == nplanes * nx * ny
            out.shape = (nplanes, nx, ny)
        else:
            out = np.empty((nplanes, nx, ny), dtype=np.float32)

        # Retrieve plane extrema
        plane_min = intstream[3 : (3 + nplanes)].view(np.float32)
        plane_max = intstream[(3 + nplanes) : (3 + 2 * nplanes)].view(np.float32)
        plane_delta = plane_max - plane_min

        # Get a 3D view of the integer stream for re-scaling
        int_field = intstream[(3 + 2 * nplanes) :].view()
        int_field.shape = (nplanes, nx, ny)

        dequantize_kernel(
            int_field,
            out,
            plane_min,
            plane_delta,
            self.nbits,
            self.transform == "Lorenzo",
        )

        return out

    def get_config(self) -> dict[str, Any]:
        config = {
            "id": self.codec_id,
            "nbits": self.nbits,
            "blosc_cname": self.blosc_cname,
            "blosc_clevel": self.blosc_clevel,
            "transform": self.transform,
        }
        if self.pow2_range:
            config["pow2_range"] = True
        return config

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> LayerQuantizer:
        return cls(**config)


numcodecs.registry.register_codec(LayerQuantizer)
