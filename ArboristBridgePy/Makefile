all:
	python setup.py build; \

local:
	python setup.py build_clib; \
	python setup.py build_ext --inplace;

locallib:
	python setup.py build_clib;

localext:
	python setup.py build_ext --inplace;

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
