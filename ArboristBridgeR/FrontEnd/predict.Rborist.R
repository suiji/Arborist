# Copyright (C)  2012-2016   Mark Seligman
##
## This file is part of ArboristBridgeR.
##
## ArboristBridgeR is free software: you can redistribute it and/or modify it
## under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 2 of the License, or
## (at your option) any later version.
##
## ArboristBridgeR is distributed in the hope that it will be useful, but
## WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with ArboristBridgeR.  If not, see <http://www.gnu.org/licenses/>.

"predict.Rborist" <- function(object, newdata, yTest=NULL, quantVec = NULL, quantiles = !is.null(quantVec), qBin = 5000, ctgCensus = "votes", ...) {
  if (!inherits(object, "Rborist"))
    stop("object not of class Rborist")
  if (is.null(object$forest))
    stop("Forest state needed for prediction")
  if (quantiles && is.null(object$leaf))
    stop("Leaf state needed for quantile")
  if (quantiles && is.null(quantVec))
    quantVec <- DefaultQuantVec()

  PredictForest(object$forest, object$leaf, object$signature, newdata, yTest, quantVec, qBin, ctgCensus)
}


PredictForest <- function(forest, leaf, sigTrain, newdata, yTest, quantVec, qBin, ctgCensus) {
  if (is.null(forest$forestNode))
    stop("Forest nodes missing")
  if (is.null(leaf))
    stop("Leaf missing")
  if (is.null(sigTrain))
    stop("Training signature missing")

  if (!is.null(quantVec)) {
    if (any(quantVec > 1) || any(quantVec < 0))
      stop("Quantile range must be within [0,1]")
    if (any(diff(quantVec) <= 0))
      stop("Quantile range must be increasing")
  }

  if (!is.null(yTest) && nrow(newdata) != length(yTest)) {
    stop("Row counts of data and test vector must match")
  }
  
  # Checks test data for conformity with training data.
  predBlock <- PredBlock(newdata, sigTrain)
  if (inherits(leaf, "LeafReg")) {
    if (is.null(quantVec)) {
      prediction <- .Call("RcppTestReg", predBlock, forest, leaf, yTest)
    }
    else {
      prediction <- .Call("RcppTestQuant", predBlock, forest, leaf, quantVec, qBin, yTest)
    }
  }
  else if (inherits(leaf, "LeafCtg")) {
    if (!is.null(quantVec))
      stop("Quantiles supported for regression case only")

    if (ctgCensus == "votes") {
      prediction <- .Call("RcppTestVotes", predBlock, forest, leaf, yTest)
    }
    else if (ctgCensus == "prob") {
      prediction <- .Call("RcppTestProb", predBlock, forest, leaf, yTest)
    }
    else {
      stop(paste("Unrecognized ctgCensus type:  ", ctgCensus))
    }
  }
  else {
    stop("Unsupported leaf type")
  }

  prediction
}
