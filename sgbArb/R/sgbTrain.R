# Copyright (C)  2012-2023   Mark Seligman
##
## This file is part of mixedSGB.
##
## mixedSGB is free software: you can redistribute it and/or modify it
## under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 2 of the License, or
## (at your option) any later version.
##
## mixedSGB is distributed in the hope that it will be useful, but
## WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with ArboristR.  If not, see <http://www.gnu.org/licenses/>.
#
#
# Checks argument semantics and initializes state for deep call.
#

sgbTrain <- function(preFormat, sampler, y, ...) UseMethod("sgbTrain")

sgbTrain.default <- function(preFormat, sampler, y,
                autoCompress = 0.25,
                maxLeaf = 0,
                minInfo = 0.01,
                minNode = if (is.factor(y)) 2 else 3,
                nLevel = 6,
                nThread = 0,
                nTree = 500,
                nu = 0.1,
                predFixed = 0,
                predProb = 0.0,
                predWeight = NULL, 
                regMono = NULL,
                splitQuant = NULL,
                thinLeaves = FALSE,
                treeBlock = 1,
                verbose = FALSE,
                ...) {
            argTrain <- mget(names(formals()), sys.frame(sys.nframe()))

    if (length(y) != preFormat$nRow)
        stop("Nonconforming design matrix and response")
            
    if (minNode < 1)
        stop("Minimum node population must be positive")
            
    if (minNode > sampler$nSamp)
        warning("Minimum node population width exceeds sample count.")

    if (maxLeaf > sampler$nSamp)
        warning("Specified leaf maximum exceeds number of samples.")

    if (nThread < 0)
        stop("Thread count must be nonnegative")
    if (nLevel < 0)
        stop("Level count must be nonnegative")
    
  # Argument checking:

    if (autoCompress < 0.0 || autoCompress > 1.0)
        stop("Autocompression plurality must be a percentage.")
    
    nPred <- length(preFormat$signature$predForm)
    if (is.null(regMono)) {
        regMono <- rep(0.0, nPred)
    }
    if (length(regMono) != nPred)
        stop("Monotonicity specifier length must match predictor count.")
    if (any(abs(regMono) > 1.0))
        stop("Monotonicity specifier contains invalid probability values.")

    if (is.null(splitQuant)) {
        splitQuant <- rep(0.5, nPred)
    }
    if (length(splitQuant) != nPred)
        stop("Split quantile specification differs from predictor count.")
    if (any(splitQuant > 1) || any(splitQuant < 0))
        stop("Split specification contains invalid quantile values.")

    if (maxLeaf < 0)
        stop("Leaf maximum must be nonnegative.")
    
  # Predictor weight constraints
    if (is.null(predWeight)) {
        predWeight <- rep(1.0, nPred)
    }
    if (length(predWeight) != nPred)
        stop("Length of predictor weight does not equal number of columns")
    if (any(predWeight < 0))
        stop("Negative predictor weights")
    if (all(predWeight == 0))
        stop("All predictor weights zero")
  # TODO:  Ensure all pred weights are numeric
  

    if (predProb != 0.0 && predFixed != 0)
      stop("Conflicting sampling specifications:  Bernoulli vs. fixed.")
    if (length(predProb) > 1)
        stop("'predProb' must have a scalar value")
    if (length(predFixed) > 1)
        stop("'predFixed' must have a scalar value")

    if (predFixed == 0) {
        predFixed <- ifelse(predProb != 0.0, 0, ifelse(nPred >= 16, 0, ifelse(!is.factor(y), max(floor(nPred/3), 1), floor(sqrt(nPred)))))
    }
    if (predProb == 0.0) {
        predProb <- ifelse(predFixed != 0, 0.0, ifelse(!is.factor(y), 0.4, ceiling(sqrt(nPred))/nPred))
    }
    if (predProb < 0 || predProb > 1.0)
        stop("'predProb' value must lie in [0,1]")
    if (predFixed < 0 || predFixed > nPred)
        stop("'predFixed' must be positive integer <= predictor count.")

    # Normalizes vector of pointwise predictor probabilites.
    meanWeight <- if (predProb == 0.0) 1.0 else predProb
    argTrain$probVec <- predWeight * (nPred * meanWeight) / sum(predWeight)
    argTrain$predWeight <- NULL
    argTrain$predProb <- NULL

    # Replaces predictor frame with preformat summaries.
    # Updates argument list with new or recomputed parameters.
    argTrain$independentTrees <- FALSE
    if (is.factor(y)) {
        argTrain$nodeScore = "logOdds"
        argTrain$loss = "logistic"
        argTrain$forestScore = "logistic"
    }
    else {
        argTrain$nodeScore = "mean"
        argTrain$loss = "l2"
        argTrain$forestScore = "sum"
    }
            
    argTrain$predFixed <- predFixed
    argTrain$splitQuant <- splitQuant
    argTrain$regMono <- regMono
    argTrain$enableCoproc <- FALSE
    argTrain$pvtBlock <- 8
    argTrain$version <- as.character(packageVersion("sgbArb"))
    if (is.factor(y)) {
        argTrain$classWeight <- c(0.0, 1.0)
    }

    trainOut <- tryCatch(.Call("trainSGB", preFormat, sampler, argTrain), error = function(e){stop(e)})

    if (verbose)
        print("Training completed.")
    trainOut
}
