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

"predict.rfArb" <- function(object,
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
  if (!inherits(object, "rfArb"))
    stop("object not of class rfArb")
  forest <- object$forest
  if (is.null(forest))
    stop("Forest state needed for prediction")
  if (is.null(object$sampler))
    stop("Sampler state needed for prediction")
  if (is.null(object$signature))
    stop("Training signature missing")
  if (nThread < 0)
    stop("Thread count must be nonnegative")
  if (is.null(forest$node))
      stop("Forest nodes missing")
  if (!is.null(yTest) && nrow(newdata) != length(yTest)) {
    stop("Test vector must conform with observations")
  }

  argPredict <- list(
      bagging = bagging,
      impPermute = 0,
      ctgProb = ctgProbabilities(object$sampler, ctgCensus),
      quantVec = getQuantiles(quantiles, object$sampler, quantVec),
      trapUnobserved = trapUnobserved,
      nThread = nThread,
      verbose = verbose)
  summaryPredict <- predictCommon(object, object$sampler, newdata, yTest, argPredict)

  if (!is.null(yTest)) { # Validation (test) included.
      c(summaryPredict$prediction, summaryPredict$validation)
  }
  else {
      summaryPredict$prediction
  }
}


ctgProbabilities <- function(sampler, ctgCensus) {
    if (is.factor(sampler$yTrain) && ctgCensus == "prob") {
        TRUE
    }
    else {
        FALSE
    }
}


# Uses quartiles by default.
#
getQuantiles <- function(quantiles, sampler, quantVec) {
    if (!is.null(quantVec)) {
        if (any(quantVec > 1) || any(quantVec < 0))
            stop("Quantile range must be within [0,1]")
        if (any(diff(quantVec) <= 0))
            stop("Quantile range must be increasing")
        quantVec
    }
    else if (!quantiles) {
        NULL
    }
    else if (is.factor(sampler$yTrain)) {
        warning("Quantiles not supported for classifcation:  ignoring")
        NULL
    }
    else {
        seq(0.25, 1.0, by = 0.25)
    }
}


# Glue-layer entry for prediction.
predictCommon <- function(objTrain, sampler, newdata, yTest, argList) {
    deframeNew <- deframe(newdata, objTrain$signature)
    tryCatch(.Call("predictRcpp", deframeNew, objTrain, sampler, yTest, argList), error = function(e) {stop(e)})
}
