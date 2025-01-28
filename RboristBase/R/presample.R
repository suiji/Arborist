# Copyright (C)  2012-2025   Mark Seligman
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
                              samplingWeight = numeric(0),
                              nSamp = 0,
                              nRep = 500,
                              withRepl = TRUE,
                              nHoldout = 0,
                              nFold = 1,
                              verbose = FALSE,
                              nTree = 0,
                              ...) {
    if (nTree != 0) {
        warning("'nTree' is deprecated in favor of 'nRep':  ignoring")
    }

    naSet <- which(is.na(y))
    if (length(naSet) != 0) {
        warning("Missing reponse indices held out from sampling.")
    }
    
    nObs <- length(y)
    if (length(naSet) >= nObs)
        stop("Missing values equal or exceed number of observations")

    if (length(naSet) + nHoldout >= nObs) {
        warning("Total held out equal or exceed number observations:  reverting 'nHoldout' to zero")
        nHoldout <- 0
    }
    

    if (nSamp < 0) {
        warning("Sample count must be nonnegative:  resetting to default")
        nSamp = 0
    }
    else if (!withRepl && nSamp > nObs) {
        warning("Sample count exceeds observation count:  resetting to default")
        nSamp = 0
    }
    
    if (length(samplingWeight) > 0) {
        ignoreWeight <- FALSE
        if (length(samplingWeight) != nObs) {
            warning("Sample weight length must match row count:  ignoring")
            ignoreWeight <- TRUE
        }
        if (all(samplingWeight == 0)) {
            warning("No nonzero weights:  ignoring")
            ignoreWeight <- TRUE
        }
        if (any(samplingWeight < 0)) {
            warning("Negative sample weights not permitted:  ignoring")
            ignoreWeight <- TRUE
        }
        if (!withRepl && sum(which(samplingWeight > 0)) < nSamp) {
            warning("Insufficiently many samples with nonzero probability:  ignoring")
            ignroreWeight <- TRUE
        }
        
        if (ignoreWeight)
            samplingWeight <- numeric(0)
    }

    ps <- presampleCommon(y, samplingWeight, nSamp, nRep, withRepl, nHoldout, nFold, naSet)
    if (verbose)
        print("Sampling completed")

    ps
}



# Glue-layer interface to sampler.
presampleCommon <- function(y, samplingWeight, nSamp, nRep, withRepl, nHoldout, nFold, naSet) {
    tryCatch(.Call("rootSample", y, samplingWeight, nSamp, nRep, withRepl, nHoldout, nFold, naSet), error = function(e){stop(e)})
}
