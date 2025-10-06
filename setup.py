from setuptools import setup

setup(
    name='layerquantizer',
    version='0.3b.1',
    description='LayerQuantizer, a numcodec-compatible quantizer/compressor for multi-layer data',
    url='https://github.com/csubich/layerquantizer',
    author='Christopher Subich',
    author_email='christopher.subich@ec.gc.ca',
    license='Apache 2.0',
    packages=['layerquantizer'],
    install_requires=['numba',
                      'numpy',
                      'numcodecs',
                      ],

    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3',
    ],
)
