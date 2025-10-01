from .layerquantizer import LayerQuantizer
import numcodecs

numcodecs.registry.register_codec(LayerQuantizer)
