# Copyright (C)  2012-2022   Mark Seligman
##
## This file is part of ArboristR.
##
## ArboristR is free software: you can redistribute it and/or modify it
## under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 2 of the License, or
## (at your option) any later version.
##
## ArboristR is distributed in the hope that it will be useful, but
## WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with ArboristR.  If not, see <http://www.gnu.org/licenses/>.

"predict.Rborist" <- function(object,
                              newdata,
                              yTest=NULL,
                              quantVec = NULL,
                              quantiles = !is.null(quantVec),
                              ctgCensus = "votes",
                              trapUnobserved = FALSE,
                              bagging = FALSE,
                              nThread = 0,
                              verbose = FALSE,
                              ...) {
  if (!inherits(object, "Rborist"))
    stop("object not of class Rborist")
  if (is.null(object$forest))
    stop("Forest state needed for prediction")
  if (is.null(object$sampler))
    stop("Sampler state needed for prediction")
  if (is.null(object$signature))
    stop("Training signature missing")
  if (nThread < 0)
    stop("Thread count must be nonnegative")
  
  if (quantiles && is.null(quantVec))
    quantVec <- defaultQuantVec()

  summaryPredict <- predictDeep(object, object$sampler, newdata, yTest, quantVec, ctgCensus, trapUnobserved, bagging, nThread, verbose)

  if (!is.null(yTest)) { # Validation (test) included.
      predictOut <- c(summaryPredict$prediction, summaryPredict$validation)
  }
  else {
      predictOut <- summaryPredict$prediction
  }
  predictOut
}


# Uses quartiles by default.
#
defaultQuantVec <- function() {
  seq(0.25, 1.0, by = 0.25)
}


predictDeep <- function(objTrain, sampler, newdata, yTest, quantVec, ctgCensus, bagging, trapUnobserved, nThread, verbose) {
  forest <- objTrain$forest

  if (is.null(forest$node))
    stop("Forest nodes missing")
  if (!is.null(quantVec)) {
    if (any(quantVec > 1) || any(quantVec < 0))
      stop("Quantile range must be within [0,1]")
    if (any(diff(quantVec) <= 0))
      stop("Quantile range must be increasing")
  }

  if (!is.null(yTest) && nrow(newdata) != length(yTest)) {
    stop("Test vector must conform with observations")
  }

  if (verbose)
      print("Beginning prediction")
  
  # Checks test data for conformity with training data.
  sigTrain <- objTrain$signature
  deframeNew <- deframe(newdata, sigTrain)

  yTrain <- sampler$yTrain
  if (is.numeric(yTrain)) {
    if (is.null(quantVec)) {
      prediction <- tryCatch(.Call("testReg", deframeNew, objTrain, sampler, yTest, trapUnobserved, bagging, nThread), error = function(e) {stop(e)})
    }
    else {
      prediction <- tryCatch(.Call("testQuant", deframeNew, objTrain, sampler, quantVec, yTest, trapUnobserved, bagging, nThread), error = function(e) {stop(e)})
    }
  }
  else if (is.factor(yTrain)) {
    if (!is.null(quantVec))
      stop("Quantiles not supported for classifcation")

    if (ctgCensus == "votes") {
      prediction <- tryCatch(.Call("testVotes", deframeNew, objTrain, sampler, yTest, trapUnobserved, bagging, nThread), error = function(e) {stop(e)})
    }
    else if (ctgCensus == "prob") {
      prediction <- tryCatch(.Call("testProb", deframeNew, objTrain, sampler, yTest, trapUnobserved, bagging, nThread), error = function(e) {stop(e)})
    }
    else if (ctgCensus == "probSample") {
        stop("Sample weighting NYI")
    }
    else {
      stop(paste("Unrecognized ctgCensus type:  ", ctgCensus))
    }
  }
  else {
    stop("Unsupported response type")
  }

  if (verbose)
      print("Prediction completed")

  prediction
}
