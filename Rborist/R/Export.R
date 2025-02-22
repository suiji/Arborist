# Copyright (C)  2012-2025  Mark Seligman
##
## This file is part of AboristR.
##
## PrimR is free software: you can redistribute it and/or modify it
## under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 2 of the License, or
## (at your option) any later version.
##
## PrimR is distributed in the hope that it will be useful, but
## WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with PrimR.  If not, see <http://www.gnu.org/licenses/>.

Export <- function(arbOut) UseMethod("Export")

Export.default <- function(arbOut) {
  warning("Export is being deprecated.  Please invoke 'expandfe' instead.");
  return (tryCatch(.Call("expandTrainRcpp", arbOut), error = function(e) {stop(e)}))
}
