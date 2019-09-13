
## Arborist: Parallelized, Extensible Random Forests


[![License](https://img.shields.io/badge/core-MPL--2-brightgreen.svg)](https://www.mozilla.org/en-US/MPL/2.0/) 
[![R License](http://img.shields.io/badge/R_Bridge-GPL%20%28%3E=%202%29-brightgreen.svg?style=flat)](http://www.gnu.org/licenses/gpl-2.0.html)
[![Python License](http://img.shields.io/badge/Python__Bridge-MIT-brightgreen.svg?style=flat)](https://opensource.org/licenses/MIT
)
[![CRAN](http://www.r-pkg.org/badges/version/Rborist)](https://cran.rstudio.com/web/packages/Rborist/index.html)
[![Downloads](http://cranlogs.r-pkg.org/badges/Rborist?color=brightgreen)](http://www.r-pkg.org/pkg/Rborist)
[![PyPI version](https://badge.fury.io/py/pyborist.svg)](https://pypi.python.org/pypi/pyborist/) 
[![Travis-CI Build Status](https://travis-ci.org/suiji/Arborist.svg?branch=master)](https://travis-ci.org/suiji/Arborist)




The Arborist provides a fast, open-source implementation of Leo Brieman's Random Forest algorithm. The Arborist achieves its speed through efficient C++ code and parallel, distributed tree construction. 

Bindings are available for [R](https://cran.r-project.org/web/packages/Rborist/index.html) and **Python (evolving)**


### R

The *Arborist* is available on CRAN as the [Rborist](https://cran.r-project.org/web/packages/Rborist/index.html) package. 

Installation of Release Version:

    > install.packages('Rborist')

Installation of Development Version:

    > ./Rborist/Package/Rborist.CRAN.sh
    > R CMD INSTALL Rborist_*.*-*.tar.gz


#### Notes
- Rborist version 0.2-2 passes all 12 CRAN builds.

#### Known issues in 0.2-2
- An autocompression error yields low-quality predictions for two-valued numerical predictors with categorical response.  A suitable workaround is to turn off autocompression by specifying "autoCompress = 1.0" at training.

### Python

 - Version 0.1-0 has been archived.
 - Version 0.2-0 is under active development.
 - Test cases sought.

### Performance 

Performance metrics have been measured using [benchm-ml](https://github.com/szilard/benchm-ml). Partial results can be found [here](https://github.com/szilard/benchm-ml/tree/master/z-other-tools)

This paper compares several implementations of the Random Forest algorithm, including the Arborist: (https://www.jstatsoft.org/article/view/v077i01/v77i01.pdf).  Benchmarks used in the study are also provided at 
https://www.jstatsoft.org/article/view/v077i01.
    
### References

- [Scalability Issues in Training Decision Trees (video)](https://www.youtube.com/watch?v=ol0SZ2Omq7w), Nimbix Developer Summit, 2017.
- [Controlling for Monotonicity in Random Forest Regressors (PDF)](http://past.rinfinance.com/agenda/2016/talk/MarkSeligman.pdf), R in Finance, May 2016.
- [ Accelerating the Random Forest algorithm for commodity parallel hardware (Video)](https://www.youtube.com/watch?v=dRZrYdhNUec), PyData, July, 2015.
- [The Arborist:  A High-Performance Random Forest (TM) Implementation](http://past.rinfinance.com/agenda/2015/talk/MarkSeligman.pdf), R in Finance, May 2015.
- [Training Random Forests on the GPU:  Tree Unrolling (PDF)](http://on-demand.gputechconf.com/gtc/2015/posters/GTC_2015_Machine_Learning___Deep_Learning_03_P5282_WEB.pdf), GTC, March, 2015.


### News/Changes
- New option 'nThread' enables specification of OpenMP thread count.
- New option 'oob' constrains prediction to the out-of-bag set, essential for variable importance testing.
- Improved memory footprint.

Correctness and runtime errors are addressed as received.  With reproducible test cases, repairs are typically uploaded to GitHub within several days.

Feature requests are addressed on a case-by-case basis.

