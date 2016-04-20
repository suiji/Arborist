
Arborist: Parallelized, Extensible Random Forests
====

[![Core License](http://img.shields.io/badge/license-MPL--2+-brightgreen.svg?style=flat)](https://www.mozilla.org/en-US/MPL/2.0/) 
[![License](http://img.shields.io/badge/license-GPL%20%28%3E=%202%29-brightgreen.svg?style=flat)](http://www.gnu.org/licenses/gpl-2.0.html)
[![CRAN](http://www.r-pkg.org/badges/version/Rborist)](https://cran.rstudio.com/web/packages/Rborist/index.html) 
[![Downloads](http://cranlogs.r-pkg.org/badges/Rborist?color=brightgreen)](http://www.r-pkg.org/pkg/Rborist)
[![PyPI version](https://badge.fury.io/py/pyborist.svg)](https://pypi.python.org/pypi/pyborist/)


The Arborist provides a fast, open-source implementation of Leo Brieman's Random Forest algorithm. The Arborist achieves its speed through efficient C++ code and parallel, distributed tree construction. 

Bindings are available for [R](https://cran.r-project.org/web/packages/Rborist/index.html) and **Python (coming soon)**


R
----

The *Arborist* is available on CRAN as the [Rborist](https://cran.r-project.org/web/packages/Rborist/index.html) package. 

- Installation of Release Version:

    > install.packages('Rborist')


- Installation of Development Version:

    # -tk


### Notes
- Version 0.x has been tested on multicore implementations 
- Version 1.0 will not be released to CRAN until vignettes, tests and (improved) examples are complete.


Python
----

The *Arborist* is will soon be available on PyPI.

    
References
----
 

News/Changes
----
- Several major refactorings have taken place since the initial CRAN release.  These were aimed not only at improving performance but also away from the Arborist's original narrow focus on high observation count and high predictor probability.  Additionally, the groundwork has been laid for more advanced features to appear in upcoming releases, such as support for heterogeneous parallelization and for sparse training sets.
- A "preTrain" feature has been added to cache initital training state.  This will save computation under iterative training schemes, such as are facilitated by the R package Caret.
- Optional vector "regMono" has been introduced to specify monotonic constraints on numeric variables under regression.
- Training now computes and saves full leaf information by default. This facilitates quantile prediction, as well as interaction with other packages, such as "forestFloor", by eliminating the need for customized training.
- Restaging and splitting now parallelize across predictor/node pairs, improving core occupancy.  For an interesting illustration of the limits to the benefits of restaging, see the RangeR preprint by Wright and Ziegler.

- Several performance issues have been resolved, but dynamic load-balancing will likely not receive further attention until version 2.
Correctness errors are being addressed as they are received.

