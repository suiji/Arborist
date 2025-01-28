# Copyright (C)  2012-2025   Mark Seligman
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

forestWeight <- function(objTrain, prediction, sampler, ...) UseMethod("forestWeight")

forestWeight.default <- function(objTrain,
                                 prediction,
                                 sampler = objTrain$sampler,
                                 nThread = 0,
                                 verbose = FALSE,
                              ...) {
  if (is.null(objTrain$forest)) {
    stop("Trained forest required for weighting")
  }

  if (is.null(objTrain$leaf)) {
    stop("Leaf information required for weighting")
  }
  else if (length(objTrain$leaf$index)==0) {
    stop("Leaf summaries required:  retrain with 'thinLeaves=FALSE'");
  }
  
  if (is.null(prediction)) {
    stop("Prediction summary required for weighting")
  }

  if (!("indices" %in% names(prediction))) {
    stop("'indices' attribute not present:  retrain with 0.3-3 or later.")
  }
  
  if (length(prediction$indices) == 0) {
    stop("Prediction indices required:  rerun prediction with 'indexing=TRUE'")
  }
  
  if (nThread < 0)
      stop("Thread count must be nonnegative")

  if (inherits(objTrain, "rfArb")) {
    sampler <- objTrain$sampler
    if (is.null(sampler)) {
      stop("Sampler required for weighting")
    }
  }
  else if (inherits(objTrain, "trainArb")) {
    if (sampler$hash != objTrain$samplerHash) {
      stop("Passed and training samplers do not match")
    }
  }
  else {
    stop("Unrecognized training class");
  }

  argList <- list(verbose = verbose, nThread=nThread)
  tryCatch(.Call("forestWeightRcpp", objTrain, sampler, prediction, argList), error = function(e) { stop(e) })
}
