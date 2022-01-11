#/usr/bin/bash

# Builds a CRAN-style temporary directory tree suitable for package
# construction, then archives, then deletes the source tree.

mkdir Rborist
cd Rborist; mkdir src; mkdir R; mkdir man; mkdir inst; cd ..
cp ../LICENSE Rborist/
cp ../R/DESCRIPTION Rborist/
cp ../R/NAMESPACE Rborist/
cp ../R/*.Rd Rborist/man/
cp ../R/NEWS Rborist/inst/
cp ../R/*R Rborist/R/
cp ../../deframeR/*.R Rborist/R/
cp ../src/*.cc Rborist/src/
cp ../src/*.h Rborist/src/
cp ../src/rf/*.cc Rborist/src/
cp ../src/rf/*.h Rborist/src/
cp ../src/forest/*.cc Rborist/src/
cp ../src/forest/*.h Rborist/src/
cp ../src/callback/*.cc Rborist/src/
cp ../src/callback/*.h Rborist/src/
cp ../../deframeR/*.cc Rborist/src/
cp ../../deframeR/*.h Rborist/src/
cp ../../deframe/*.cc Rborist/src/
cp ../../deframe/*.h Rborist/src/
cp ../../cart/*.cc Rborist/src/
cp ../../cart/*.h Rborist/src/
cp ../../core/*.cc Rborist/src/
cp ../../core/*.h Rborist/src/
cp ../../forest/*.cc Rborist/src/
cp ../../forest/*.h Rborist/src/
cp ../../forest/bridge/*.cc Rborist/src/
cp ../../forest/bridge/*.h Rborist/src/
cp ../../obs/*.cc Rborist/src/
cp ../../obs/*.h Rborist/src/
cp ../../frontier/*.cc Rborist/src/
cp ../../frontier/*.h Rborist/src/
cp ../../rf/*.cc Rborist/src/
cp ../../rf/*.h Rborist/src/
cp ../../split/*.cc Rborist/src/
cp ../../split/*.h Rborist/src/
cp -r ../tests Rborist
cp -r ../vignettes Rborist
