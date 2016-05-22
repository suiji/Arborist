from distutils import log
from distutils.command.build_clib import build_clib as _build_clib
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext as _build_ext
import numpy as np
from os import listdir, path



pyx_src_dir = 'pyborist'
cc_src_dir = path.join('..', 'ArboristCore')


all_pyx_files = [x for x in listdir(pyx_src_dir) if x.endswith('.pyx')]
all_cpp_core_files = [path.join(cc_src_dir, x) 
    for x in listdir(cc_src_dir) if x.endswith('.cc')] + \
    [path.join(pyx_src_dir, 'callback.cc')]


lib_aborist_core = ('libaboristcore', 
    {
        'sources': all_cpp_core_files,
        'include_dirs': [cc_src_dir, pyx_src_dir]
    }
)


extensions = [
    Extension(x[:-4], [path.join(pyx_src_dir, x)],
        language = 'c++',
        include_dirs = [pyx_src_dir, cc_src_dir, np.get_include()],
        library_dirs = [pyx_src_dir, cc_src_dir]
    ) for x in all_pyx_files
]



#TODO
# msvc may raise this error if '/openmp' flag is added:
# error C3016: index variable in OpenMP 'for' statement must have signed integral type
extra_compile_args = {
    'msvc': ['/Ox', '/fp:fast'], 
    'mingw32': ['-std=c++11', '-fopenmp','-O3','-ffast-math'],
    'unix': ['-std=c++11', '-fopenmp','-O3','-ffast-math']
}
extra_link_args = {
    'mingw32': ['-fopenmp'],
    'unix': ['-fopenmp']
}


class build_clib(_build_clib):
    def build_libraries(self, libraries):
        """
        The default version does not accept extra compile args!!!
        So the hack is needed here. 
        Most code are exactly the same but the compiler.compile() gets injected.
        """
        compiler_type = self.compiler.compiler_type
        compile_args = []
        if compiler_type in extra_compile_args:
            compile_args = extra_compile_args[compiler_type]

        for (lib_name, build_info) in libraries:
            sources = list(build_info.get('sources'))
            log.info("building '%s' library", lib_name)
            macros = build_info.get('macros')
            include_dirs = build_info.get('include_dirs')
            objects = self.compiler.compile(sources,
                                            output_dir=self.build_temp,
                                            macros=macros,
                                            include_dirs=include_dirs,
                                            extra_postargs=compile_args, # important
                                            debug=self.debug)
            self.compiler.create_static_lib(objects, lib_name,
                                            output_dir=self.build_clib,
                                            debug=self.debug)


class build_ext(_build_ext):
    def build_extensions(self):
        """
        http://stackoverflow.com/questions/724664/python-distutils-how-to-get-a-compiler-that-is-going-to-be-used

        We want to inject the compile / link args based on different compilers.
        """
        compiler_type = self.compiler.compiler_type
        if compiler_type in extra_compile_args:
           for ext in self.extensions:
               ext.extra_compile_args = extra_compile_args[compiler_type]
        if compiler_type in extra_link_args:
            for ext in self.extensions:
                ext.extra_link_args = extra_link_args[compiler_type]
        super(build_ext, self).build_extensions()

    def get_ext_fullpath(self, ext_name):
        """
        The default one returns os.path.join(package_dir, filename)
        But we want to move the files to the real subdir if --inplace here
        """
        parsed_ext_path = super(build_ext, self).get_ext_fullpath(ext_name)
        package_dir = path.dirname(parsed_ext_path)
        filename = path.basename(parsed_ext_path)
        real_ext_path = path.join(package_dir, pyx_src_dir, filename)
        return real_ext_path



setup(
    name = 'pyborist',
    version = '0.0.1',
    description = 'a Random Forest library',
    packages = ['pyborist'],
    libraries = [lib_aborist_core],
    cmdclass = {'build_clib': build_clib, 'build_ext': build_ext},
    ext_modules = cythonize(extensions)
)
