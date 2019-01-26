#/usr/bin/bash

# Alternate package-building script employs a hacked Makevars
# for Travis.

mkdir Rborist
cd Rborist; mkdir src; mkdir R; mkdir man; mkdir inst; cd ..
cp ../LICENSE Rborist/
cp ../FrontEnd/DESCRIPTION Rborist/
cp ../FrontEnd/NAMESPACE Rborist/
cp ../FrontEnd/*.Rd Rborist/man/
cp ../FrontEnd/NEWS Rborist/inst/
cp ../FrontEnd/*R Rborist/R/
cp -r ../tests Rborist
cp -r ../vignettes Rborist

cp ../Shared/Makevars.travis Rborist/src/Makevars
cp ../Shared/*.cc Rborist/src/
cp ../Shared/*.h Rborist/src/
cp ../../ArboristCore/*.cc Rborist/src/
cp ../../ArboristCore/*.h Rborist/src/

R CMD build Rborist --no-build-vignettes
rm -rf Rborist

