# Copyright (C)  2012-2024   Mark Seligman
##
## This file is part of Rborist.
##
## Rborist is free software: you can redistribute it and/or modify it
## under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 2 of the License, or
## (at your option) any later version.
##
## Rborist is distributed in the hope that it will be useful, but
## WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with Rborist.  If not, see <http://www.gnu.org/licenses/>.

"predict.arbTrain" <- function(object,
                              newdata,
                              sampler = NULL,
                              yTest=NULL,
                              keyedFrame = FALSE,
                              quantVec = NULL,
                              quantiles = !is.null(quantVec),
                              ctgCensus = "votes",
                              indexing = FALSE,
                              trapUnobserved = FALSE,
                              bagging = FALSE,
                              nThread = 0,
                              verbose = FALSE,
                              ...) {
  if (inherits(object, "rfArb")) {
      sampler <- object$sampler
  }
  else {
    if (is.null(sampler))
        stop("Sampler state needed for prediction")
    if (sampler$hash != object$samplerHash)
        stop("Sampler hashes do not match.")
  }

  forest <- object$forest
  if (is.null(forest))
    stop("Forest state needed for prediction")
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
      ctgProb = ctgProbabilities(sampler, ctgCensus),
      quantVec = getQuantiles(quantiles, sampler, quantVec),
      indexing = indexing,
      trapUnobserved = trapUnobserved,
      nThread = nThread,
      verbose = verbose)
  summaryPredict <- predictCommon(object, sampler, newdata, yTest, keyedFrame, argPredict)

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


# Glue-layer entry for prediction.
predictCommon <- function(object, sampler, newdata, yTest, keyedFrame, argList) {
    deframeNew <- deframe(newdata, object$signature, keyedFrame)
    tryCatch(.Call("predictRcpp", deframeNew, object, sampler, yTest, argList), error = function(e) {stop(e)})
}
