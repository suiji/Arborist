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

Streamline <- function(arbOut) {
    UseMethod("Streamline.rfArb")
}


Streamline.rfArb <- function(arbOut) {
  warning("Streamline is being deprecated.  Please train with 'thinLeaves' instead.");
  if (!inherits(arbOut, "rfArb"))
    stop("object not of class rfArb")
  if (is.null(arbOut$sampler))
    stop("Sampler state needed for prediction")
  rb <- arbOut
  rb$sampler <- list(
      yTrain = arbOut$sampler$yTrain,
      nSamp = arbOut$sampler$nSamp,
      nTree = arbOut$sampler$nTree,
      samples = raw(0),
      hash = arbOut$sampler$hash
  )

  rb
}
