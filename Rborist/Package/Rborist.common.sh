#/usr/bin/bash

# Builds a CRAN-style temporary directory tree suitable for package
# construction, then archives, then deletes the source tree.

mkdir Rborist
cd Rborist; mkdir src; mkdir R; mkdir man; mkdir inst; cd ..
cp ../LICENSE Rborist/
cp ../DESCRIPTION Rborist/
cp ../NAMESPACE Rborist/
cp ../R/*.Rd Rborist/man/
cp ../R/NEWS Rborist/inst/
cp ../R/*R Rborist/R/
cp ../../deframeR/*.R Rborist/R/
cp ../src/*.{cc,h} Rborist/src/
cp ../../RboristBase/src/*.{cc,h} Rborist/src/
cp ../../RboristBase/R/*.R Rborist/R/
cp ../../RboristBase/R/*.Rd Rborist/man/
cp ../../deframeR/*.{cc,h} Rborist/src/
cp ../../deframe/*.{cc,h} Rborist/src/
cp ../../cart/*.{cc,h} Rborist/src/
cp ../../core/*.{cc,h} Rborist/src/
cp ../../forest/*.{cc,h} Rborist/src/
cp ../../forest/bridge/*.{cc,h} Rborist/src/
cp ../../obs/*.{cc,h} Rborist/src/
cp ../../frontier/*.{cc,h} Rborist/src/
cp ../../rf/*.{cc,h} Rborist/src/
cp ../../split/*.{cc,h} Rborist/src/
cp -r ../tests Rborist
cp -r ../vignettes Rborist
