
## Arborist: Parallelized, Extensible Decision Tree Tools


[![License](https://img.shields.io/badge/core-MPL--2-brightgreen.svg)](https://www.mozilla.org/en-US/MPL/2.0/) 
[![R License](http://img.shields.io/badge/R_Bridge-GPL%20%28%3E=%202%29-brightgreen.svg?style=flat)](http://www.gnu.org/licenses/gpl-2.0.html)
[![Python License](http://img.shields.io/badge/Python__Bridge-MIT-brightgreen.svg?style=flat)](https://opensource.org/licenses/MIT
)
[![CRAN](http://www.r-pkg.org/badges/version/Rborist)](https://cran.rstudio.com/web/packages/Rborist/index.html)
[![Downloads](http://cranlogs.r-pkg.org/badges/Rborist?color=brightgreen)](http://www.r-pkg.org/pkg/Rborist)
[![PyPI version](https://badge.fury.io/py/pyborist.svg)](https://pypi.python.org/pypi/pyborist/) 
[![Travis-CI Build Status](https://travis-ci.org/suiji/Arborist.svg?branch=master)](https://travis-ci.org/suiji/Arborist)




The Arborist project hosts fast, open-source implementations of several decision-tree algorithms.  Breiman and Cutler's **Random Forest** algorithm is implemented and Friedman's **Stochastic Gradient Boosting** is available as an alpha release.  A spin providing Friedman and Fisher's **PRIM** ("Patient Rule Induction Method") has been developed by Decision Patterns, LLC.  Arborist derivatives achieve their speed through parallelized and vectorized inner loops.  Parallel, distributed training is also possible for independently-trained trees. Considerable attention has been devoted to minimizing and regularizing data movement, a key challenge to accelerating these algorithms.

Bindings are provided for [R](https://cran.r-project.org/web/packages/Rborist/index.html).  A language-agnostic bridge design supports development of bindings for additional front ends, such as **Python** and **Julia**.

### R

**CRAN** hosts the released package [Rborist](https://cran.r-project.org/web/packages/Rborist/index.html), which implements the Random Forest algorithm.

Installation of the *released* version using **R**:

    > install.packages('Rborist')

Installation of the *development* version, hosted on this archive, from the top-level directory:

    > ./Rborist/Package/Rborist.CRAN.sh
    > R CMD INSTALL Rborist_*.*-*.tar.gz

A **CRAN**-friendly snapshot of the *development* source is mirrored by the neighboring archive [Rborist.CRAN](https://github.com/suiji/Rborist.CRAN).  This archive is intended for remote access by **R** utilities such as *devtools*.

#### Notes
- **Rborist** version 0.3-5 has been submitted to **CRAN**.

### Python

 - Version 0.1-0 has been archived.
 - Test cases sought.

### Performance 

Performance metrics have been measured using [benchm-ml](https://github.com/szilard/benchm-ml). Partial results can be found [here](https://github.com/szilard/benchm-ml/tree/master/z-other-tools)

Some users have reported diminished performance when running single-threaded.  We recommend running with at least two cores, as frequently-executed inner loops have been cast specifically to take advantage of multiple cores.  In particular, when using a scaffold such as __caret__, please prefer to let Rborist be greedier with cores than is the scaffold.

This paper compares several implementations of the Random Forest algorithm, including Rborist: (https://www.jstatsoft.org/article/view/v077i01/v77i01.pdf).  Benchmarks used in the study are also provided at 
https://www.jstatsoft.org/article/view/v077i01.
    
A recent paper compares several categories of regression tools, including Random Forests.  Rborist is among the faster packages offering high prediction accuracy: (https://doi.org/10.1109/ACCESS.2019.2933261).  Based on the findings, we have updated the package's default settings.  In particular, fixed-number predictor sampling (__mtry__) appears to provide more accurate predictions at low dimension than the current approach of Bernoulli sampling.
    
### References

- [Scalability Issues in Training Decision Trees (video)](https://www.youtube.com/watch?v=ol0SZ2Omq7w), Nimbix Developer Summit, 2017.
- [Controlling for Monotonicity in Random Forest Regressors (PDF)](http://past.rinfinance.com/agenda/2016/talk/MarkSeligman.pdf), R in Finance, May 2016.
- [ Accelerating the Random Forest algorithm for commodity parallel hardware (Video)](https://www.youtube.com/watch?v=dRZrYdhNUec), PyData, July, 2015.
- [The Arborist:  A High-Performance Random Forest (TM) Implementation](http://past.rinfinance.com/agenda/2015/talk/MarkSeligman.pdf), R in Finance, May 2015.
- [Training Random Forests on the GPU:  Tree Unrolling (PDF)](http://on-demand.gputechconf.com/gtc/2015/posters/GTC_2015_Machine_Learning___Deep_Learning_03_P5282_WEB.pdf), GTC, March, 2015.


### News/Changes
- New archive [sgbArb.CRAN](https://github.com/suiji/sgbArb.CRAN) mirrors the **sgbArb** package for stochastic gradient boosting.
- New archive [Rborist.CRAN](https://github.com/suiji/Rborist.CRAN) mirrors the **Rborist** package source in a form directly amenable to utilities such as *devtools*.
- New command *rfTrain* exposes the training component of the compound format/sample/train/validate task performed by *rfArb*.  This provides separate training of sampled, prefomatted data.
- New prediction option *keyedFrame* accesses prediction columns by name, bypassing a previous requirement that training and prediction frames have the same column ordering.  In addition to arbitrary ordering, the prediction frame may now include columns not submitted to training.
- New command *forestWeight* computes Meinshausen's forest-wide weights.  Nonterminals are weighted in addition to leaves, both to facilitate post-pruning and to accommodate early exit under prediction with trap-and-bail.
- New prediction option *indexing=TRUE* records final node indices of tree walks.
- Training ignores missing predictor values, splitting over appropriately reduced subnodes.
- Quantile estimation supports both leaf and nonterminal (i.e., trap-and-bail) prediction modes.
- Prediction and validiation support large (> 32 bits) observation counts.
- Support for training more than 2^32 observations may be enabled by recompiling.
- New option *impPermute* introduces permutation-based variable importance.

### Known issues
 - Following the introduction of support for standalone sampling, a break in backward compatibility appears in versions 0.3-0 and higher of the *Rborist* package.  Prediction with models trained using earlier versions throws an unidentified-index exception from within the *Rcpp* glue layer.  Older models should therefore be retrained in order to use version 0.3-0 and above.

Correctness and runtime errors are addressed as received.  With reproducible test cases, repairs are typically uploaded to GitHub within several days.

Feature requests are addressed on a case-by-case basis.

