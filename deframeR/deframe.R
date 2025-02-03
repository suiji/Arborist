# Copyright (C)  2012-2025  Mark Seligman
##
## This file is part of deframeR.
##
## deframeR is free software: you can redistribute it and/or modify it
## under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 2 of the License, or
## (at your option) any later version.
##
## deframeR is distributed in the hope that it will be useful, but
## WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with deframeR.  If not, see <http://www.gnu.org/licenses/>.

# Groups predictors into like-typed blocks and creates zero-based type
# summaries.
#

deframe <- function(x, sigTrain = NULL, keyed = FALSE, nThread = 0) {
  threadsUsed <- tryCatch(.Call("setThreadCount", nThread))

  # Argument checking:
  # For now, only numeric and unordered factor types supported.
  #
  # For now, RLE frame is ranked on both training and prediction.
  if (is.data.frame(x)) {
      data.table::setDTthreads(threadsUsed)
      dt <- data.table::setDT(x)[,tryCatch(.Call("columnOrder", x, sigTrain, keyed))]
      colSurvey <- sapply(dt, function(col) ifelse(is.numeric(col) || (is.factor(col) && !is.ordered(col)), TRUE, FALSE))
      if (length(which(colSurvey)) != ncol(dt)) {
          stop("Frame columns must be either numeric or unordered factor")
      }
      classVec <- sapply(dt, function(col) class(col))
      ret <- tryCatch(.Call("deframeDF", dt, classVec, lapply(dt, levels)[classVec == "factor"], lapply(dt, factor)[classVec == "factor"], sigTrain), error = function(e) {stop(e)} )
  }
  else {
    if (keyed) {
        warning("Keyed access not yet supported for matrix types:  ignoring.")
    }
    if (inherits(x, "dgCMatrix")) {
      ret <- tryCatch(.Call("deframeIP", x), error= print)
    }
    else if (is.matrix(x)) {
      if (is.integer(x) && is.factor(x) && !is.ordered(x)) {
        ret <- tryCatch(.Call("deframeFac", data.matrix(x) ), error=function(e) {stop(e)} )
      }
      else if (is.integer(x)) {
        ret <- tryCatch(.Call("deframeNum", data.matrix(x) ), error=function(e) {stop(e)} )
      }
      else if (is.numeric(x)) {
        ret <- tryCatch(.Call("deframeNum", x), error=function(e) {stop(e)})
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

  dummy <- tryCatch(.Call("setThreadCount", 0))
  ret
}
