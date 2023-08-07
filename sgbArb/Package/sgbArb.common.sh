#/usr/bin/bash

# Builds a CRAN-style temporary directory tree suitable for package
# construction, then archives, then deletes the source tree.

mkdir sgbArb
cd sgbArb; mkdir src; mkdir R; mkdir man; mkdir inst; cd ..
cp ../LICENSE sgbArb/
cp ../DESCRIPTION sgbArb/
cp ../NAMESPACE sgbArb/
cp ../R/*.Rd sgbArb/man/
cp ../R/NEWS sgbArb/inst/
cp ../R/*R sgbArb/R/
cp ../../deframeR/*.R sgbArb/R/
cp ../src/*.{cc,h} sgbArb/src/
cp ../../RboristBase/src/*.{cc,h} sgbArb/src/
cp ../../RboristBase/R/*.R sgbArb/R/
cp ../../RboristBase/R/*.Rd sgbArb/man/
cp ../../deframeR/*.{cc,h} sgbArb/src/
cp ../../deframe/*.{cc,h} sgbArb/src/
cp ../../cart/*.{cc,h} sgbArb/src/
cp ../../core/*.{cc,h} sgbArb/src/
cp ../../forest/*.{cc,h} sgbArb/src/
cp ../../forest/bridge/*.{cc,h} sgbArb/src/
cp ../../obs/*.{cc,h} sgbArb/src/
cp ../../frontier/*.{cc,h} sgbArb/src/
cp ../../sgb/*.{cc,h} sgbArb/src/
cp ../../split/*.{cc,h} sgbArb/src/
cp -r ../tests sgbArb
cp -r ../vignettes sgbArb
