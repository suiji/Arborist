all:
	export CFLAGS='-std=c++11'; \
	python setup.py build; \

local:
	export CFLAGS='-std=c++11'; \
	python setup.py build_clib; \
	python setup.py build_ext --inplace; \
	mv -f *.pyd pyborist/ 2>/dev/null; \
	mv -f *.so pyborist/ 2>/dev/null;
	mv -f *.dynlib pyborist/ 2>/dev/null; 

locallib:
	export CFLAGS='-std=c++11'; \
	python setup.py build_clib; \

localext:
	export CFLAGS='-std=c++11'; \
	python setup.py build_ext --inplace; \
	mv -f *.pyd pyborist/ 2>/dev/null; \
	mv -f *.so pyborist/ 2>/dev/null;
	mv -f *.dynlib pyborist/ 2>/dev/null; 

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
	rm -f *.dynlib; \
