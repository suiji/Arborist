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

deframe <- function(x, sigTrain = NULL) {
  # Argument checking:
  if (any(is.na(x))) {
    stop("NA not supported in design matrix")
  }

  # For now, only numeric and unordered factor types supported.
  #
  # For now, RLE frame is ranked on both training and prediction.
  if (is.data.frame(x)) {
      dt <- data.table::setDT(x)
      colSurvey <- sapply(dt, function(col) ifelse(is.numeric(col) || (is.factor(col) && !is.ordered(col)), TRUE, FALSE))
      if (length(which(colSurvey)) != ncol(dt)) {
          stop("Frame columns must be either numeric or unordered factor")
      }
      predForm <- sapply(dt, function(col) ifelse(is.numeric(col), "numeric", "factor"))
      hasFactor <- sapply(dt, function(col) ifelse(is.numeric(col), FALSE, TRUE))
      return(tryCatch(.Call("DeframeDF", dt, predForm, lapply(dt, levels)[hasFactor], lapply(dt, factor)[hasFactor], sigTrain), error = function(e) {stop(e)} ))
  }
  else if (inherits(x, "dgCMatrix")) {
     return(tryCatch(.Call("DeframeIP", x), error= print))
  }
  else if (is.matrix(x)) {
    if (is.integer(x) && is.factor(x) && !is.ordered(x)) {
      return(tryCatch(.Call("DeframeFac", data.matrix(x) ), error=function(e) {stop(e)} ))
    }
    else if (is.integer(x)) {
      warning("Integer matrix values intepreted as numeric");
      return(tryCatch(.Call("DeframeNum", data.matrix(x) ), error=function(e) {stop(e)} ))
    }
    else if (is.numeric(x)) {
      return(tryCatch(.Call("DeframeNum", x), error=function(e) {stop(e)}))
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
