from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Iterable, Any, TYPE_CHECKING

from .layerquantizer import LayerQuantizer

if TYPE_CHECKING:
    from zarr.abc.codec import ArrayBytesCodec
    from zarr.core.buffer import Buffer, NDBuffer, default_buffer_prototype
    from zarr.core.array_spec import ArraySpec
    from zarr.core.chunk_grids import ChunkGrid
else:
    try:
        from zarr.abc.codec import ArrayBytesCodec
        from zarr.core.buffer import Buffer, NDBuffer, default_buffer_prototype
        from zarr.core.array_spec import ArraySpec
        from zarr.core.chunk_grids import ChunkGrid
    except ImportError:
        # Fallback for environments where zarr 3 is not installed
        class ArrayBytesCodec:
            pass

        class Buffer:
            pass

        class NDBuffer:
            pass

        class ArraySpec:
            pass

        class ChunkGrid:
            pass

@dataclass(frozen=True)
class LayerQuantizerCodec(ArrayBytesCodec):
    nbits: int = 16
    transform: str = "Lorenzo"
    pow2_range: bool = False
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LayerQuantizerCodec:
        # Zarr 3 might pass the whole dict or just the configuration part
        config = data.get("configuration", data)
        return cls(
            nbits=config.get("nbits", 16),
            transform=config.get("transform", "Lorenzo"),
            pow2_range=config.get("pow2_range", False),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": "layerquantizer0.3",
            "configuration": {
                "nbits": self.nbits,
                "transform": self.transform,
                "pow2_range": self.pow2_range,
            }
        }

    async def encode(
        self, chunks_and_specs: Iterable[tuple[NDBuffer | None, ArraySpec]]
    ) -> Iterable[Buffer | None]:
        # Create a reusable LayerQuantizer instance (not for compression, just for config)
        lq = LayerQuantizer(
            nbits=self.nbits,
            transform=self.transform,
            pow2_range=self.pow2_range
        )
        prototype = default_buffer_prototype()
        
        out: list[Buffer | None] = []
        for chunk, spec in chunks_and_specs:
            if chunk is None:
                out.append(None)
                continue
            
            arr = chunk.as_numpy_array()
            if self.nbits > 24:
                # Just convert to bytes.  Numpy's tobytes() uses native order;
                # for float32 we should probably ensure a specific order if Zarr 3 requires it,
                # but here we just follow the input's lead or assume native is fine if it's
                # just a pass-through. However, the plan says Little Endian.
                out.append(prototype.buffer.from_bytes(arr.astype('<f4').tobytes()))
            else:
                encoded_int32 = lq._encode_to_int32(arr)
                out.append(prototype.buffer.from_bytes(encoded_int32.astype('<i4').tobytes()))
        return out

    async def decode(
        self, chunks_and_specs: Iterable[tuple[Buffer | None, ArraySpec]]
    ) -> Iterable[NDBuffer | None]:
        lq = LayerQuantizer(
            nbits=self.nbits,
            transform=self.transform,
            pow2_range=self.pow2_range
        )
        prototype = default_buffer_prototype()
        
        out: list[NDBuffer | None] = []
        for chunk, spec in chunks_and_specs:
            if chunk is None:
                out.append(None)
                continue
            
            b = chunk.to_bytes()
            if self.nbits > 24:
                decoded_arr = np.frombuffer(b, dtype='<f4').reshape(spec.shape)
            else:
                intstream = np.frombuffer(b, dtype='<i4')
                decoded_arr = lq._decode_from_int32(intstream)
                # Reshape back to original chunk shape
                decoded_arr = decoded_arr.reshape(spec.shape)
            
            out.append(prototype.nd_buffer.from_numpy_array(decoded_arr.astype(np.float32)))
        return out

    def compute_encoded_size(self, input_byte_length: int, chunk_spec: ArraySpec) -> int:
        if self.nbits > 24:
            return input_byte_length
        
        # input_byte_length is size in bytes of float32 array.
        # Number of elements N = input_byte_length / 4
        # For 2D planes, we need to know the number of planes.
        # chunk_spec.shape gives the shape of the chunk.
        
        shape = chunk_spec.shape
        if len(shape) < 2:
             # Should have been validated, but let's be safe
             return (3 + 2 * 1 + (input_byte_length // 4)) * 4
        
        nplanes = 1
        for d in shape[:-2]:
            nplanes *= d
        
        # Size in bytes: (3 + 2*nplanes + size) * 4
        return (3 + 2 * nplanes + (input_byte_length // 4)) * 4

    def validate(
        self, *, shape: tuple[int, ...], dtype: Any, chunk_grid: ChunkGrid
    ) -> None:
        # Verify float32. Handles numpy dtype and Zarr 3 DataType objects
        valid_dtype = False
        if dtype == np.float32:
            valid_dtype = True
        elif hasattr(dtype, "name") and dtype.name == "float32":
             valid_dtype = True
        elif str(dtype).startswith("float32") or str(dtype).startswith("Float32"):
             valid_dtype = True
        
        if not valid_dtype:
            # LayerQuantizer strictly requires float32
            raise ValueError(f"LayerQuantizer requires float32, got {dtype!r}")
        
        if len(shape) < 2:
            # Need at least 2 dimensions for the 2D plane logic
            raise ValueError(f"LayerQuantizer requires at least 2 dimensions, got {len(shape)}")

    def resolve_metadata(self, chunk_spec: ArraySpec) -> ArraySpec:
        return chunk_spec
