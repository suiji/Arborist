# Copyright (C)  2012-2023   Mark Seligman
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
## along with Rborist.  If not, see <http://www.gnu.org/licenses/>.

# Uses quartiles by default.
#
getQuantiles <- function(quantiles, sampler, quantVec) {
    if (!is.null(quantVec)) {
        if (any(quantVec > 1) || any(quantVec < 0))
            stop("Quantile range must be within [0,1]")
        if (any(diff(quantVec) <= 0))
            stop("Quantile range must be increasing")
        quantVec
    }
    else if (!quantiles) {
        NULL
    }
    else if (is.factor(sampler$yTrain)) {
        warning("Quantiles not supported for classifcation:  ignoring")
        NULL
    }
    else {
        seq(0.25, 1.0, by = 0.25)
    }
}
