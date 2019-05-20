#/usr/bin/bash

# Builds a CRAN-style temporary directory tree suitable for package
# construction, then archives, then deletes the source tree.

mkdir Rborist
cd Rborist; mkdir src; mkdir R; mkdir man; mkdir inst; cd ..
cp ../LICENSE Rborist/
cp ../FrontEnd/DESCRIPTION Rborist/
cp ../FrontEnd/NAMESPACE Rborist/
cp ../FrontEnd/*.Rd Rborist/man/
cp ../FrontEnd/NEWS Rborist/inst/
cp ../FrontEnd/*R Rborist/R/
cp ../Shared/*.cc Rborist/src/
cp ../Shared/*.h Rborist/src/
cp ../../framemapR/*.cc Rborist/src/
cp ../../framemapR/*.h Rborist/src/
cp ../../framemap/*.cc Rborist/src/
cp ../../framemap/*.h Rborist/src/
cp ../../core/*.cc Rborist/src/
cp ../../core/*.h Rborist/src/
cp ../../coreRf/*.cc Rborist/src/
cp ../../coreRf/*.h Rborist/src/
cp -r ../tests Rborist
cp -r ../vignettes Rborist
