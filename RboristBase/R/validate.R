# Copyright (C)  2012-2024   Mark Seligman
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

validate <- function(train, preFormat, ...) UseMethod("validate")

validate.default <- function(train,
                             preFormat,
                             sampler = NULL,
                             ctgCensus = "votes",
                             impPermute = 0,
                             quantVec = NULL,
                             quantiles = !is.null(quantVec),
                             indexing = FALSE,
                             trapUnobserved = FALSE,
                             nThread = 0,
                             verbose = FALSE,
                             ...) {
  if (!inherits(train, "arbTrain"))
    stop("Unrecognized training object")

  if (is.null(train$forest)) {
    stop("Trained forest required for validation")
  }

  if (is.null(preFormat)) {
    stop("Preformatted frame required for validation")
  }
  
  if (is.null(sampler)) {
    sampler <- train$sampler
  }
  if (is.null(sampler)) {
    stop("Sampler required for validation")
  }

  if (nThread < 0)
      stop("Thread count must be nonnegative")
  if (is.null(preFormat) && impPermute > 0)
      stop("Pre-formatted observation set required for permutation testing.")

  if (impPermute < 0) {
      warning("Negative permutation count:  substituting zero.")
      impPermute <- 0
  }
        

  argPredict <- list(
      bagging = TRUE,
      impPermute = impPermute,
      ctgProb = ctgProbabilities(sampler, ctgCensus),
      quantVec = getQuantiles(quantiles, sampler, quantVec),
      indexing = indexing,
      trapUnobserved = trapUnobserved,
      nThread = nThread,
      verbose = verbose)
  validateCommon(train, sampler, preFormat, argPredict)
}


# Glue-layer entry for validation.
validateCommon <- function(objTrain, sampler, preFormat, argList) {
  tryCatch(.Call("validateRcpp", preFormat, objTrain, sampler, argList), error = function(e) { stop(e) })
}
