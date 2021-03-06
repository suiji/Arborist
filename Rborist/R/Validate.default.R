# Copyright (C)  2012-2020   Mark Seligman
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
"Validate.default" <- function(preFormat, train, y, ctgCensus = "votes",
                               impPermute = 0, quantVec = NULL,
                               quantiles = !is.null(quantVec),
                               nThread = 0, verbose = FALSE) {
  if (is.null(train$bag)) {
    stop("Bag required for verification")
  }
  if (is.null(train$forest)) {
    stop("Trained forest required for verification")
  }
  if (is.null(train$leaf)) {
    stop("Leaf information required for verification")
  }
  if (nThread < 0)
    stop("Thread count must be nonnegative")

  ValidateDeep(preFormat, train, y, impPermute, ctgCensus, quantVec, quantiles, nThread, verbose)
}


ValidateDeep <- function(preFormat, objTrain, y, impPermute, ctgCensus, quantVec, quantiles, nThread, verbose) {
  if (is.factor(y)) {
    if (ctgCensus == "votes") {
        if (verbose)
            print("Validation:  census only");
        validation <- tryCatch(.Call("ValidateVotes", preFormat, objTrain, y, impPermute, nThread), error = function(e) { stop(e) })
    }
    else if (ctgCensus == "prob") {
        if (verbose)
            print("Validation:  categorical probabilities");
        validation <- tryCatch(.Call("ValidateProb", preFormat, objTrain, y, impPermute, nThread), error = function(e) { stop(e) })
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
          quantVec <- DefaultQuantVec()
        }
        validation <- tryCatch(.Call("ValidateQuant", preFormat, objTrain, y, impPermute, quantVec, nThread), error = function(e) { stop(e) })
    }
    else {
        if (verbose)
            print("Validation:  ordinary regression");
        validation <- tryCatch(.Call("ValidateReg", preFormat, objTrain, y, impPermute, nThread), error = function(e) { stop(e) })
    }
  }

  if (verbose)
      print("Validation complete")
  
  validation
}
