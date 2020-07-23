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

# Groups predictors into like-typed blocks and creates zero-based type
# summaries.
#

presort <- function(x, predFrame) {
    if (is.data.frame(x)) {
        preFormat <- list(
            predFrame = predFrame,
            summaryRLE = tryCatch(.Call("PresortDF", data.table::setDT(x))),
            obsHash = digest::digest(x)
        )
    }
    else {
        preFormat <- list(
            predFrame = predFrame,
            summaryRLE = tryCatch(.Call("PresortNum", predFrame)),
            obsHash = digest::digest(x)
        )
    }
    class(preFormat) <- "PreFormat"
    return(preFormat)
}


deframe <- function(x, sigTrain = NULL) {
  # Argument checking:
  if (any(is.na(x))) {
    stop("NA not supported in design matrix")
  }

  # For now, only numeric and unordered factor types supported.
  #
  if (is.data.frame(x)) {
      dt <- data.table::setDT(x)
      colSurvey <- sapply(dt, function(col) ifelse(is.numeric(col) || (is.factor(col) && !is.ordered(col)), TRUE, FALSE))
      if (length(which(colSurvey)) != ncol(dt)) {
          stop("Frame columns must be either numeric or unordered factor")
      }
      predForm <- sapply(dt, function(col) ifelse(is.numeric(col), "numeric", "factor"))
      lv <- lapply(dt, levels) # All string levels, regardless whether realized.
      xFac <- data.matrix(Filter(function(col) ifelse(is.numeric(col), FALSE, TRUE), dt)) - 1
      hasFactor <- sapply(dt, function(col) ifelse(is.numeric(col), FALSE, TRUE))
      # If already trained, extract factors and reconcile:
      if (!is.null(sigTrain) && any(hasFactor)) {
          xFac <- tryCatch(.Call("FrameReconcile", xFac, predForm, lv[hasFactor], sigTrain), error = function(e) {stop(e)} )
      }
      # As with xFac, xNum should be reconstructed internally for validation.
      xNum <- data.matrix(Filter(function(col) ifelse(is.numeric(col), TRUE, FALSE), dt))
      codes <- lapply(dt, factor) # Realized levels only.
      predFrame <- tryCatch(.Call("WrapFrame", dt, xNum, xFac, predForm, lv[hasFactor], codes[hasFactor]), error = function(e) {stop(e)} )
  }
  else if (inherits(x, "dgCMatrix")) {
     predFrame <- tryCatch(.Call("WrapSparse", x), error= print)
  }
  else if (is.matrix(x)) {
    if (is.integer(x)) {
      predFrame <- tryCatch(.Call("WrapNum", data.matrix(x)), error=function(e) {stop(e)} )
    }
    else if (is.numeric(x)) {
      predFrame <- tryCatch(.Call("WrapNum", x), error=function(e) {stop(e)})
    }
    else if (is.character(x)) {
      stop("Character data not yet supported")
    }
    else {
      stop("Unsupported matrix type")
    }
  }
  else {
    stop("Expecting data frame or matrix")
  }

    if (is.null(sigTrain)) {
        return(presort(x, predFrame));
    }
    else {
        return(predFrame);
    }
}
