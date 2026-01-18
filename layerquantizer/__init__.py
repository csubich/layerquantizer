from .layerquantizer import LayerQuantizer
from .zarr3_codec import LayerQuantizerCodec as LayerQuantizerCodec
import numcodecs

numcodecs.registry.register_codec(LayerQuantizer)
