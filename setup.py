from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="layerquantizer",
    version="0.3.2",
    description="LayerQuantizer, a numcodec-compatible quantizer/compressor for multi-layer data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/csubich/layerquantizer",
    author="Christopher Subich",
    author_email="christopher.subich@ec.gc.ca",
    license="Apache 2.0",
    packages=["layerquantizer"],
    install_requires=[
        "numba",
        "numpy",
        "numcodecs",
    ],
    python_requires=">=3.11",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
    ],
    entry_points={
        "zarr.codecs": [
            "layerquantizer0.3 = layerquantizer.zarr3_codec:LayerQuantizerCodec",
        ],
        "numcodecs.codecs": [
            "layerquantizer0.3b = layerquantizer.layerquantizer:LayerQuantizer",
        ],
    },
)
