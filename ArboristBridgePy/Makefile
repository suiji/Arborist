all:
	export CFLAGS='-std=c++11'; \
	python setup.py build; \

local:
	export CFLAGS='-std=c++11'; \
	python setup.py build_clib; \
	python setup.py build_ext --inplace; \
	python -c 'from pyborist.utils import move_dynamic_files;move_dynamic_files()'

locallib:
	export CFLAGS='-std=c++11'; \
	python setup.py build_clib;

localext:
	export CFLAGS='-std=c++11'; \
	python setup.py build_ext --inplace; \
	python -c 'from pyborist.utils import move_dynamic_files;move_dynamic_files()'

clean:
	rm -rf pyborist/build; \
	rm -rf pyborist/__pycache__; \
	rm -f pyborist/*.cpp; \
	rm -f pyborist/*.c; \
	rm -f pyborist/*.pyd; \
	rm -f pyborist/*.so; \
	rm -f pyborist/*.dynlib; \
	rm -rf build/; \
	rm -rf __pycache__/; \
	rm -f *.pyd; \
	rm -f *.so;
	rm -f *.dynlib;
