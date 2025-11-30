from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

# Define the Cython extensions
extensions = [
    Extension(
        "qqs",
        ["qqs.pyx"],
        include_dirs=[numpy.get_include()],
        language="c++",
        extra_compile_args=["-O3", "-ffast-math"],
        extra_link_args=["-O3"],
    )
]

# TODO: Add other dependencies and make this complete
setup(
    name="phlag",
    version="0.1.0",
    description="Phlag flags phlogenetic anomalies across the genome.",
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "language_level": 3,
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
            "embedsignature": True,
        },
    ),
    zip_safe=False,
    python_requires=">=3.9",
    install_requires=["cython", "numpy", "treeswift"],
)
