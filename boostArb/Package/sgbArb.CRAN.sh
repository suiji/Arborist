#/usr/bin/bash

# Builds a CRAN-style temporary directory tree suitable for package
# construction, then archives, then deletes the source tree.

bash sgbArb.common.sh

cp ../R/Makevars sgbArb/src/
cp ../R/Makevars.win sgbArb/src/

R CMD build sgbArb
rm -rf sgbArb

