#/usr/bin/bash

# Alternate package-building script employs a hacked Makevars
# for Travis.

bash Rborist.common.sh

cp ../Shared/Makevars.travis Rborist/src/Makevars

R CMD build Rborist --no-build-vignettes
rm -rf Rborist

