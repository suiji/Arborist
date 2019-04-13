# Copyright (C)  2012-2018   Mark Seligman
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
                             quantVec = NULL, quantiles = !is.null(quantVec),
                             qBin = 5000, nThread = 0, verbose = FALSE) {
  if (is.null(preFormat$predBlock)) {
    stop("Pre-formatted observations required for verification")
  }
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

  ValidateDeep(preFormat$predBlock, train, y, ctgCensus, quantVec, quantiles, qBin, nThread, verbose)
}


ValidateDeep <- function(predBlock, objTrain, y, ctgCensus, quantVec, quantiles, qBin, nThread, verbose) {
  if (verbose)
      print("Beginning validation");
  if (is.factor(y)) {
    if (ctgCensus == "votes") {
      validation <- tryCatch(.Call("ValidateVotes", predBlock, objTrain, y, nThread), error = function(e) { stop(e) })
    }
    else if (ctgCensus == "prob") {
      validation <- tryCatch(.Call("ValidateProb", predBlock, objTrain, y, nThread), error = function(e) { stop(e) })
    }
    else {
      stop(paste("Unrecognized ctgCensus type:  ", ctgCensus))
    }
  }
  else {
    if (quantiles) {
      if (is.null(quantVec)) {
        quantVec <- DefaultQuantVec()
      }
      validation <- tryCatch(.Call("ValidateQuant", predBlock, objTrain, y, quantVec, qBin, nThread), error = function(e) { stop(e) })
    }
    else {
      validation <- tryCatch(.Call("ValidateReg", predBlock, objTrain, y, nThread), error = function(e) { stop(e) })
    }
  }

  if (verbose)
      print("Validation complete")
  
  validation
}
