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

# Pre-formats a data frame or buffer, if not already pre-formatted.
# If already pre-formatted, verifies types of member fields.
PreFormat.default <- function(x, verbose = FALSE) {
    if (inherits(x, "PreFormat")) {
        preFormat <- x
        if (!inherits(preFormat$predFrame, "Frame")) {
            stop("Missing Frame")
        }
        if (!inherits(preFormat$summaryRLE, "RLEFrame")) {
            stop("Missing RLEFrame")
        }
        if (verbose)
            print("Training set already pre-formatted")
    }
    else {
        if (verbose)
            print("Blocking frame")

        predFrame <- PredFrame(x)

        if (verbose)
            print("Pre-sorting")
        preFormat <- list(
            predFrame = predFrame,
            summaryRLE = .Call("Presort", predFrame),
            obsHash = digest::digest(x)
        )
        class(preFormat) <- "PreFormat"
        if (verbose)
            print("Pre-formatting completed")
    }

    preFormat
}


# Groups predictors into like-typed blocks and creates zero-based type
# summaries.
#
PredFrame <- function(x, sigTrain = NULL) {
  # Argument checking:
  if (any(is.na(x))) {
    stop("NA not supported in design matrix")
  }

  # For now, only numeric and factor types supported.
  #
  if (is.data.frame(x)) {
      dt <- data.table::setDT(x)
      xFac <- data.matrix(Filter(function(col) ifelse(is.factor(col) && !is.ordered(col), TRUE, FALSE), dt)) - 1
      xNum <- data.matrix(Filter(function(col) ifelse(is.numeric(col), TRUE, FALSE), dt))
      if (ncol(xNum) + ncol(xFac) != ncol(dt)) {
          stop("Frame column with unsupported data type")
      }
      lv <- sapply(dt, levels)
      colCard <- sapply(dt, function(col) ifelse(is.numeric(col), 0, length(levels(col))))
      predMap <- c(which(colCard == 0), which(colCard != 0)) - 1
      if (!is.null(sigTrain) && any(colCard != 0)) {
          xFac <- tryCatch(.Call("FrameReconcile", xFac, predMap, lv[colCard != 0], sigTrain), error = function(e) {stop(e)} )
      }      

      return(tryCatch(.Call("WrapFrame", dt, xNum, xFac, predMap, colCard[colCard != 0], lv[colCard != 0]), error = function(e) {stop(e)} ))
  }
  else if (inherits(x, "dgCMatrix")) {
     return(tryCatch(.Call("FrameSparse", x), error= print))
  }
  else if (is.matrix(x)) {
    if (is.integer(x)) {
      return(tryCatch(.Call("FrameNum", data.matrix(x)), error=function(e) {stop(e)} ))
    }
    else if (is.numeric(x)) {
      return(tryCatch(.Call("FrameNum", x), error=function(e) {stop(e)}))
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
}
