#/usr/bin/bash

# Builds a CRAN-style temporary directory tree suitable for package
# construction, then archives, then deletes the source tree.

bash sgbArb.common.sh

cp ../R/Makevars sgbArb/src/

R CMD build sgbArb --no-build-vignettes
rm -rf sgbArb

