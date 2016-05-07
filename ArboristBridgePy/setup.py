from os import listdir, path

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

pyx_src = 'src'
cpp_lib = path.join('..', 'ArboristCore')

all_pyx_files = [x for x in listdir(pyx_src) if x.endswith('.pyx')]


# cyaa.pyx -> ('cyaa', ['cyaa.pyx', 'cpp_lib/aa.cc'])
extensions = [
    Extension(x[:-4],
        [path.join(pyx_src, x), path.join(cpp_lib, x[2:-4]+'.cc')],
        language = 'c++',
        include_dirs = [pyx_src, cpp_lib],
        #libraries = [...], #TODO what?
        library_dirs = [pyx_src, cpp_lib]
    ) for x in all_pyx_files
]


setup(ext_modules = cythonize(extensions))
