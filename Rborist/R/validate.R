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

validate <- function(train, ...) UseMethod("validate")


validate.default <- function(train,
                             sampler,
                             preFormat = NULL,
                             ctgCensus = "votes",
                             impPermute = 0,
                             quantVec = NULL,
                             quantiles = !is.null(quantVec),
                             trapUnobserved = FALSE,
                             nThread = 0,
                             verbose = FALSE) {
  if (is.null(sampler)) {
    stop("Sampler required for validation")
  }
  if (is.null(train$forest)) {
    stop("Trained forest required for validation")
  }
  if (is.null(train$leaf)) {
    stop("Leaf information required for validation")
  }
  if (nThread < 0)
      stop("Thread count must be nonnegative")
  if (is.null(preFormat) && impPermute > 0)
      stop("Pre-formatted observation set required for permutation testing.")

  validateDeep(train, preFormat, sampler, impPermute, ctgCensus, quantVec, quantiles, trapUnobserved, nThread, verbose)
}


validateDeep <- function(objTrain, preFormat, sampler, impPermute, ctgCensus, quantVec, quantiles, trapUnobserved, nThread, verbose) {
  if (is.factor(sampler$yTrain)) {
    if (ctgCensus == "votes") {
        if (verbose)
            print("Validation:  census only");
        validation <- tryCatch(.Call("validateVotes", preFormat, objTrain, sampler, sampler$yTrain, impPermute, trapUnobserved, nThread), error = function(e) { stop(e) })
    }
    else if (ctgCensus == "prob") {
        if (verbose)
            print("Validation:  categorical probabilities");
        validation <- tryCatch(.Call("validateProb", preFormat, objTrain, sampler, sampler$yTrain, impPermute, trapUnobserved, nThread), error = function(e) { stop(e) })
    }
    else {
      stop(paste("Unrecognized ctgCensus type:  ", ctgCensus))
    }
  }
  else {
    if (quantiles) {
        if (verbose)
            print("ValidatIon:  quantiles");
        if (is.null(quantVec)) {
          quantVec <- defaultQuantVec()
        }
        validation <- tryCatch(.Call("validateQuant", preFormat, objTrain, sampler, sampler$yTrain, impPermute, quantVec, trapUnobserved, nThread), error = function(e) { stop(e) })
    }
    else {
        if (verbose)
            print("Validation:  ordinary regression");
        validation <- tryCatch(.Call("validateReg", preFormat, objTrain, sampler, sampler$yTrain, impPermute, trapUnobserved, nThread), error = function(e) { stop(e) })
    }
  }

  if (verbose)
      print("Validation complete")
  
  validation
}
