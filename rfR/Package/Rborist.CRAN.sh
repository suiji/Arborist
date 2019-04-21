#/usr/bin/bash

# Builds a CRAN-style temporary directory tree suitable for package
# construction, then archives, then deletes the source tree.

bash Rborist.common.sh

cp ../Shared/Makevars Rborist/src/
cp ../Shared/Makevars.win Rborist/src/

R CMD build Rborist
rm -rf Rborist

