Code for version 0 has been tested on multicore implementations for the R front end.

Latest changes are geared toward hierarchical parallelization:  distributed over heterogeneous.  Low-level support to be available before actual interfaces are provided.

A "pretrain" feature is being added to cache initital training state.  This will save computation under iterative training schemes, such as are facilitated by the R package Caret.  This has entailed considerable refactoring from which the dust has not quite settled.

Several performance issues have been resolved, but multicore load-balancing will likely not get much attention until version 2.
Correctness errors are being addressed as they are received.



Version 1 does not go to CRAN until examples improved and vignettes provided.
