# Copyright (C)  2012-2021   Mark Seligman
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
"Streamline.default" <- function(rs) {
  if (!inherits(rs, "Rborist"))
    stop("object not of class Rborist")
  if (is.null(rs$sampler))
    stop("Sampler state needed for prediction")
  rb <- rs
  rb$sampler <- list(
      yTrain = rs$yTrain,
      samplerBlock= raw(0),
      nTree = rs$sampler$nTree
      )

  rb
}
