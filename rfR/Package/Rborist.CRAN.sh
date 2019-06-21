#/usr/bin/bash

# Builds a CRAN-style temporary directory tree suitable for package
# construction, then archives, then deletes the source tree.

bash Rborist.common.sh

cp ../src/Makevars Rborist/src/
cp ../src/Makevars.win Rborist/src/

R CMD build Rborist
rm -rf Rborist

