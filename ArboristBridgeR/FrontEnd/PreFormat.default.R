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
PreFormat.default <- function(x) {
    if (inherits(x, "PreFormat") || inherits(x, "PreTrain")) {
        preFormat <- x
        if (!inherits(preFormat$predBlock, "PredBlock")) {
            stop("Missing PredBlock")
        }
        if (!inherits(preFormat$rowRank, "RowRank")) {
            stop("Missing RowRank")
        }
    }
    else {
        predBlock <- PredBlock(x)
        rowRank <- .Call("Presort", predBlock)

        preFormat <- list(
            predBlock = predBlock,
            rowRank = rowRank
        )
        class(preFormat) <- "PreFormat"
    }

    preFormat
}
