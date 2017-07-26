# Copyright (C)  2012-2017   Mark Seligman
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
                             qBin = 5000) {
  if (is.null(preFormat$predBlock)) {
    stop("Incomplete PreFormat object")
  }
  if (is.null(train$forest)) {
    stop("Incomplete Train object")
  }
  if (is.null(train$leaf)) {
    stop("Missing leaf information")
  }
  predBlock <- preFormat$predBlock
  forest <- train$forest
  leaf <- train$leaf
  if (is.factor(y)) {
    if (ctgCensus == "votes") {
      validation <- tryCatch(.Call("RcppValidateVotes", predBlock, forest, leaf, y), error = function(e) { stop(e) })
    }
    else if (ctgCensus == "prob") {
      validation <- tryCatch(.Call("RcppValidateProb", predBlock, forest, leaf, y), error = function(e) { stop(e) })
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
      validation <- tryCatch(.Call("RcppValidateQuant", predBlock, forest, leaf, y, quantVec, qBin), error = function(e) { stop(e) })
    }
    else {
      validation <- tryCatch(.Call("RcppValidateReg", predBlock, forest, leaf, y), error = function(e) { stop(e) })
    }
  }

  validation
}
