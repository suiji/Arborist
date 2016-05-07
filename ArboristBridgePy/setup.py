from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

extensions = [
    Extension('*', ['src/*.pyx'],
        include_dirs = ['src/', '../ArboristCore'],
        #libraries = [...], #TODO what?
        library_dirs = ['src/', '../ArboristCore']
    )
]

setup(ext_modules = cythonize(extensions))
