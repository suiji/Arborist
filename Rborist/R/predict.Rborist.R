# Copyright (C)  2012-2021   Mark Seligman
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

"predict.Rborist" <- function(object, newdata, yTest=NULL, quantVec = NULL, quantiles = !is.null(quantVec), ctgCensus = "votes", oob = FALSE, nThread = 0, verbose = FALSE, ...) {
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
    quantVec <- DefaultQuantVec()

  summaryPredict <- PredictDeep(object, newdata, yTest, quantVec, ctgCensus, oob, nThread, verbose)

  if (!is.null(yTest)) { # Validation (test) included.
      predictOut <- c(summaryPredict$prediction, summaryPredict$validation)
  }
  else {
      predictOut <- summaryPredict$prediction
  }
  predictOut
}


PredictDeep <- function(objTrain, newdata, yTest, quantVec, ctgCensus, oob, nThread, verbose) {
  forest <- objTrain$forest

  if (is.null(forest$forestNode))
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
  deframeRow <- deframe(newdata, sigTrain)

  sampler <- objTrain$sampler
  if (inherits(sampler, "SamplerReg")) {
    if (is.null(quantVec)) {
      prediction <- tryCatch(.Call("TestReg", deframeRow, objTrain, yTest, oob, nThread), error = function(e) {stop(e)})
    }
    else {
      prediction <- tryCatch(.Call("TestQuant", deframeRow, objTrain, quantVec, yTest, oob, nThread), error = function(e) {stop(e)})
    }
  }
  else if (inherits(sampler, "SamplerCtg")) {
    if (!is.null(quantVec))
      stop("Quantiles not supported for classifcation")

    if (ctgCensus == "votes") {
      prediction <- tryCatch(.Call("TestVotes", deframeRow, objTrain, yTest, oob, nThread), error = function(e) {stop(e)})
    }
    else if (ctgCensus == "prob") {
      prediction <- tryCatch(.Call("TestProb", deframeRow, objTrain, yTest, oob, nThread), error = function(e) {stop(e)})
    }
    else {
      stop(paste("Unrecognized ctgCensus type:  ", ctgCensus))
    }
  }
  else {
    stop("Unsupported sampler type")
  }

  if (verbose)
      print("Prediction completed")

  prediction
}
