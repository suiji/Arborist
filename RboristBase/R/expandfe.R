# Copyright (C)  2012-2023  Mark Seligman
##
## This file is part of RboristBase.
##
## RboristBase is free software: you can redistribute it and/or modify it
## under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 2 of the License, or
## (at your option) any later version.
##
## RboristBase is distributed in the hope that it will be useful, but
## WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with RboristBase.  If not, see <http://www.gnu.org/licenses/>.

expandfe <- function(arbOut) UseMethod("expandfe")


expandfe.default <- function(arbOut) {
  if (!inherits(arbOut, "arbTrain")) {
    stop("Expecting an arbTrain object.")
  }

  return (tryCatch(.Call("expandTrainRcpp", arbOut), error = function(e) {stop(e)}))
}
