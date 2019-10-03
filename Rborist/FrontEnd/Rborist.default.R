# Copyright (C)  2012-2019   Mark Seligman
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
#
#
# Checks argument semantics and initializes state for deep call.
#
"Rborist.default" <- function(x,
                y,
                autoCompress = 0.25,              
                ctgCensus = "votes",
                classWeight = NULL,
                maxLeaf = 0,
                minInfo = 0.01,
                minNode = ifelse(is.factor(y), 2, 3),
                nLevel = 0,
                nSamp = 0,
                nThread = 0,
                nTree = 500,
                noValidate = FALSE,
                predFixed = 0,
                predProb = 0.0,
                predWeight = NULL, 
                quantVec = NULL,
                quantiles = !is.null(quantVec),
                regMono = NULL,
                rowWeight = NULL,
                splitQuant = NULL,
                thinLeaves = ifelse(is.factor(y), TRUE, FALSE),
                treeBlock = 1,
                verbose = FALSE,
                withRepl = TRUE,
                ...) {
    argList <- mget(names(formals()), sys.frame(sys.nframe()))

    if (nTree <= 0)
        stop("Tree count must be positive")
    if (nSamp < 0)
        stop("Sample count must be nonnegative")
    if (nThread < 0)
        stop("Thread count must be nonnegative")
    if (nLevel < 0)
        stop("Level count must be nonnegative")
    
    if (any(is.na(y)))
        stop("NA not supported in response")
    if (!is.numeric(y) && !is.factor(y))
        stop("Expecting numeric or factor response")

    preFormat <- PreFormat(x, verbose)
    predFrame <- preFormat$predFrame

    
  # Argument checking:

    nRow <- predFrame$nRow
    if (length(y) != nRow)
        stop("Nonconforming design matrix and response")

    if (autoCompress < 0.0 || autoCompress > 1.0)
        stop("Autocompression plurality must be a percentage.")
    
    nPred <- length(predFrame$signature$predMap)

    if (is.null(regMono)) {
        regMono <- rep(0.0, nPred)
    }
    if (length(regMono) != nPred)
        stop("Monotonicity specifier length must match predictor count.")
    if (any(abs(regMono) > 1.0))
        stop("Monotonicity specifier contains invalid probability values.")
    if (is.factor(y) && any(regMono != 0)) {
        stop("Monotonicity undefined for categorical response")
    }

    if (is.null(splitQuant)) {
        splitQuant <- rep(0.5, nPred)
    }
    if (length(splitQuant) != nPred)
        stop("Split quantile specification differs from predictor count.")
    if (any(splitQuant > 1) || any(splitQuant < 0))
        stop("Split specification contains invalid quantile values.")


    nSamp <- ifelse(nSamp > 0, nSamp, ifelse(withRepl, nRow, round(1-exp(-1)*nRow)))

    if (maxLeaf < 0)
        stop("Leaf maximum must be nonnegative.")
    if (maxLeaf > nSamp)
        warning("Specified leaf maximum exceeds number of samples.")


  # Class weights

    nCtg <- ifelse(is.factor(y), max(as.integer(y)), 0)
    if (is.factor(y)) {
        if (!is.null(classWeight)) {
            if (is.numeric(classWeight)) {
                if (length(classWeight) != nCtg)
                    stop("class weights must conform to response cardinality")
                if (any(classWeight < 0))
                    stop("class weights must be nonnegative")
                if (all(classWeight == 0.0)) {
                    stop("class weights cannot all be zero")
                }
            }
            else if (classWeight == "balance") { # place-holder value
                classWeight <- rep(0.0, nCtg)
            }
            else {
                stop("Unrecognized class weights")
            }
        }
        else {
            classWeight <- rep(1.0, nCtg)
        }
    }
    else if (!is.null(classWeight)) {
        stop("class weights only defined for classification")
    }
  
  # Height constraints
  if (minNode < 1)
    stop("Minimum node size must be positive")
  else if (minNode > nSamp)
    stop("Minimum splitting width exceeds sample count")

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
  
  # No holes allowed in response
    if (is.factor(y) && any(table(y) == 0))
        stop("Empty classes not supported in response.")

  # Sample weight constraints.
    if (is.null(rowWeight)) {
        rowWeight <- rep(1.0, nRow)
    }

    if (length(rowWeight) != nRow) {
        stop("Sample weight length must match row count")
    }    

    if (any(rowWeight < 0)) {
        stop("Negative weights not permitted")
    }
    
  # Quantile constraints:  regression only
    if (quantiles && is.factor(y))
        stop("Quantiles supported for regression case only")
    if (quantiles && thinLeaves)
        stop("Thin leaves insufficient for validating quantiles.")
    
    if (!is.null(quantVec)) {
        if (any(quantVec > 1) || any(quantVec < 0))
            stop("Quantile range must be within [0,1]")
        if (any(diff(quantVec) <= 0))
            stop("Quantile range must be increasing")
    }

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
        stop("'predFixed' must be positive integer <= predictor count")

    # Normalizes vector of pointwise predictor probabilites.
    meanWeight <- ifelse(predProb == 0.0, 1.0, predProb)
    argList$probVec <- predWeight * (nPred * meanWeight) / sum(predWeight)
    argList$predWeight <- NULL
    argList$predProb <- NULL
    
    # Replaces predictor frame with preformat summaries.
    # Updates argument list with new or recomputed parameters.
    argList$x <- NULL
    argList$predFrame <- predFrame
    argList$summaryRLE <- preFormat$summaryRLE
    argList$nCtg <- nCtg
    argList$nSamp <- nSamp
    argList$predFixed <- predFixed
    argList$classWeight <- classWeight
    argList$rowWeight <- rowWeight
    argList$quantiles <- quantiles
    argList$quantVec <- quantVec
    argList$splitQuant <- splitQuant
    argList$regMono <- regMono
    argList$enableCoproc <- FALSE

    argList$pvtBlock <- 8

    RFDeep(argList)
}


RFDeep <- function(argList) {
    train <- tryCatch(.Call("TrainRF", argList), error = function(e){stop(e)})

    predInfo <- train[["predInfo"]]
    names(predInfo) <- argList$predFrame$colnames
    training = list(
        call = match.call(),
        info = predInfo,
        version = "0.2-3",
        diag = train[["diag"]]
    )

    if (argList$noValidate) {
        validation <- NULL
    }
    else {
        validation <- ValidateDeep(argList$predFrame, train, argList$y, argList$ctgCensus, argList$quantVec, argList$quantiles, argList$nThread, argList$verbose)
    }

    arbOut <- list(
        bag = train$bag,
        forest = train$forest,
        leaf = train$leaf,
        signature = argList$predFrame$signature,
        training = training,
        validation = validation
    )
    class(arbOut) <- "Rborist"

    arbOut
}




# Uses quartiles by default.
#
DefaultQuantVec <- function() {
  seq(0.25, 1.0, by = 0.25)
}
