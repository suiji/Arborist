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
## along with ArboristR.  If not, see <http://www.gnu.org/licenses/>.
#
#

presample <- function(y, ...) UseMethod("presample")


presample.default <- function(y,
                              rowWeight = NULL,
                              nSamp = 0,
                              nTree = 500,
                              withRepl = TRUE,
                              verbose = FALSE,
                              ...) {
    if (nTree <= 0)
        stop("Tree count must be positive")

    nRow <- length(y)

    if (nSamp == 0) {
        if (withRepl)
            nSamp <- nRow
        else
            nSamp <- round((1-exp(-1))*nRow)
    }
    else if (nSamp < 0)
        stop("Sample count must be nonnegative")
    else if (!withRepl && nSamp > nRow)
        stop("Sample count exceeds observation count but not replacing")
    
    if (!is.null(rowWeight)) {
        if (length(rowWeight) != nRow) {
            stop("Sample weight length must match row count")
        }
        if (all(rowWeight == 0)) {
            stop("No nonzero weights")
        }
        if (any(rowWeight < 0)) {
            stop("Negative sample weights not permitted")
        }
        if (!withRepl && sum(which(rowWeight > 0)) < nSamp)
            stop("Insufficiently many samples with nonzero probability")
    }

    ps <- presampleCommon(y, rowWeight, nSamp, nTree, withRepl)
    if (verbose)
        print("Sampling completed")

    ps
}



# Glue-layer interface to sampler.
presampleCommon <- function(y, rowWeight, nSamp, nTree, withRepl) {
    tryCatch(.Call("rootSample", y, rowWeight, nSamp, nTree, withRepl), error = function(e){stop(e)})
}
