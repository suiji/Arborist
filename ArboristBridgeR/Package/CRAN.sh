#/usr/bin/bash

# Builds a CRAN-style temporary directory tree suitable for package
#construction, then archives, then deletes the source tree.

mkdir Rborist
cd Rborist; mkdir src; mkdir R; mkdir man; cd ..
cp ../DESCRIPTION Rborist/
cp ../NAMESPACE Rborist/
cp ../LICENSE Rborist/
cp ../*.Rd Rborist/man/
cp ../*R Rborist/R/
cp ../*.cc Rborist/src/
cp ../*.h Rborist/src/
cp ../Makevars Rborist/src/
cp ../../ArboristCore/*.cc Rborist/src/
cp ../../ArboristCore/*.h Rborist/src/
cp -r ../tests Rborist
R CMD build Rborist
rm -rf Rborist

