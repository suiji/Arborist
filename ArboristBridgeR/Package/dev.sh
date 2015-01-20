#/usr/bin/bash

# Builds a CRAN-style temporary directory tree suitable for package
#construction, then archives, then deletes the source tree.

# Makefile.in does not appear to be necessary for the build, and
# configure.ac may also turn out not to be needed.

rm -r Rborist
rm Rborst.tar.gz
mkdir Rborist
cd Rborist; mkdir src; mkdir R; mkdir man; cd ..
cp ../DESCRIPTION Rborist/
cp ../NAMESPACE Rborist/
#cp ../configure.ac Rborist/
#cp ../Makefile.in Rborist/src/
cp ../Makevars Rborist
cp ../*.Rd Rborist/man/
cp ../*R Rborist/R/
cp ../LICENSE Rborist/R/
cp ../*.cc Rborist/src/
cp ../*.h Rborist/src/
cp ../Makevars Rborist/src/
cp ../../ArboristCore/*.cc Rborist/src/
cp ../../ArboristCore/*.h Rborist/src/
cp ../../ArboristCore/LICENSE Rborist/src/
cd Rborist; autoconf; cd ..
tar -czvf Rborist.tar.gz Rborist/

