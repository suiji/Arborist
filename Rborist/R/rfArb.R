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
#
#
# Checks argument semantics and initializes state for deep call.
#

rfArb <- function(x, y, ...) UseMethod("rfArb")


rfArb.default <- function(x,
                          y,
                autoCompress = 0.25,              
                ctgCensus = "votes",
                classWeight = NULL,
                impPermute = 0,
                maxLeaf = 0,
                minInfo = 0.01,
                minNode = if (is.factor(y)) 2 else 3,
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
                thinLeaves = is.factor(y),
                trapUnobserved = FALSE,
                treeBlock = 1,
                verbose = FALSE,
                withRepl = TRUE,
                ...) {
    argTrain <- mget(names(formals()), sys.frame(sys.nframe()))

    if (nTree <= 0)
        stop("Tree count must be positive")
    if (nThread < 0)
        stop("Thread count must be nonnegative")
    if (nLevel < 0)
        stop("Level count must be nonnegative")
    
    if (any(is.na(y)))
        stop("NA not supported in response")

    if (!is.numeric(y) && !is.factor(y))
        stop("Expecting numeric or factor response")

    if (impPermute < 0)
        warning("Negative permutation count:  ignoring.")

    if (impPermute > 1)
        warning("Permutation count limited to one.")
    
    if (impPermute > 0 && noValidate)
        warning("Variable importance requires validation:  ignoring")
    
    preFormat <- preformat(x, verbose)
    
  # Argument checking:

    nRow <- preFormat$nRow
    if (length(y) != nRow)
        stop("Nonconforming design matrix and response")

    if (autoCompress < 0.0 || autoCompress > 1.0)
        stop("Autocompression plurality must be a percentage.")
    
    nPred <- preFormat$signature$nPred

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

    if (maxLeaf < 0)
        stop("Leaf maximum must be nonnegative.")
    
  # Class weights
    nCtg <- if (is.factor(y)) length(levels(y)) else 0
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
        stop("Minimum node population must be positive")

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


  # Quantile constraints:  regression only
    if (quantiles && is.factor(y))
        stop("Quantiles supported for regression case only")
    if (quantiles && thinLeaves)
        stop("Thin leaves insufficient for deriving quantiles.")
    
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
        stop("'predFixed' must be positive integer <= predictor count.")

    sampler <- presample(y, rowWeight, nSamp, nTree, withRepl)

    if (minNode > sampler$nSamp)
        warning("Minimum node population width exceeds sample count.")

    if (maxLeaf > sampler$nSamp)
        warning("Specified leaf maximum exceeds number of samples.")

    # Normalizes vector of pointwise predictor probabilites.
    meanWeight <- if (predProb == 0.0) 1.0 else predProb
    argTrain$probVec <- predWeight * (nPred * meanWeight) / sum(predWeight)
    argTrain$predWeight <- NULL
    argTrain$predProb <- NULL
    
    # Replaces predictor frame with preformat summaries.
    # Updates argument list with new or recomputed parameters.
    argTrain$predFixed <- predFixed
    argTrain$classWeight <- classWeight
    argTrain$splitQuant <- splitQuant
    argTrain$regMono <- regMono
    argTrain$enableCoproc <- FALSE
    argTrain$pvtBlock <- 8

    train <- tryCatch(.Call("rfTrain", preFormat, sampler, argTrain), error = function(e){stop(e)})

    if (argTrain$noValidate) {
        summaryValidate <- NULL
    }
    else {
        argPredict <- list(
            bagging = TRUE,
            impPermute = impPermute,
            ctgProb = ctgProbabilities(sampler, ctgCensus),
            quantVec = getQuantiles(quantiles, sampler, quantVec),
            trapUnobserved = trapUnobserved,
            nThread = nThread,
            verbose = verbose)
        # can validate without prediction if permutation tests not requested:
        # summaryValidate <- validate(train$sampler, train$leaf) 
        summaryValidate <- validateCommon(train, sampler, preFormat, argPredict)
    }

    postTrain(preFormat, sampler, train, summaryValidate, impPermute)
}


postTrain <- function(preFormat, sampler, train, summaryValidate, impPermute) {
    predInfo <- train[["predInfo"]]
    names(predInfo) <- preFormat$signature$colNames
    training = list(
        call = match.call(),
        info = predInfo,
        version = "0.3-0",
        diag = train[["diag"]]
    )

    if (impPermute > 0) {
        arbOut <- list(
            sampler = sampler,
            leaf = train$leaf,
            forest = train$forest,
            predMap = train$predMap,
            signature = preFormat$signature,
            training = training,
            prediction = summaryValidate$prediction,
            validation = summaryValidate$validation,
            importance = summaryValidate$importance
        )
    }
    else {
        arbOut <- list(
            sampler = sampler,
            leaf = train$leaf,
            forest = train$forest,
            predMap = train$predMap,
            signature = preFormat$signature,
            training = training,
            prediction = summaryValidate$prediction,
            validation = summaryValidate$validation
        )
    }
    class(arbOut) <- "rfArb"

    arbOut
}

