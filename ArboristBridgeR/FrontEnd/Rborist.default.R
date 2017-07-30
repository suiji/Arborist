# Copyright (C)  2012-2017   Mark Seligman
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

# 'y' should be defined on entry.  Other entry points to be used for different behaviours.
#
"Rborist.default" <- function(x, y, nTree=500, withRepl = TRUE,
                autoCompress = 0.25,              
                ctgCensus = "votes",
                classWeight = NULL,
                maxLeaf = 0,
                minInfo = 0.01,
                minNode = ifelse(is.factor(y), 2, 3),
                nLevel = 0,
                noValidate = FALSE,
                nSamp = 0,
                predFixed = 0,
                predProb = 0.0,
                predWeight = NULL, 
                quantVec = NULL,
                quantiles = !is.null(quantVec),
                qBin = 5000,
                regMono = NULL,
                rowWeight = NULL,
                splitQuant = NULL,
                thinLeaves = FALSE,
                treeBlock = 1,
                pvtBlock = 8,
                ...) {

  # Argument checking:
  if (inherits(x, "PreTrain") || inherits(x, "PreFormat")) {
    preFormat <- x 
  }
  else {
    preFormat <- PreFormat(x)
  }
  predBlock <- preFormat$predBlock
  nPred <- predBlock$nPredNum + predBlock$nPredFac
  nRow <- predBlock$nRow

  if (length(y) != nRow)
      stop("Nonconforming design matrix and response")

  if (autoCompress < 0.0 || autoCompress > 1.0)
      stop("Autocompression plurality must be a percentage.")

  if (is.null(regMono)) {
    regMono <- rep(0.0, nPred)
  }
  else {
    if (length(regMono) != nPred)
        stop("Monotone specification length differs from predictor count.")
    if (any(abs(regMono) > 1.0))
        stop("Monotone specification contains invalid probability values.")
  }
    
  if (is.null(splitQuant)) {
    splitQuant <- rep(0.5, nPred)
  }
  else {
    if (length(splitQuant) != nPred)
        stop("Split quantile specification differs from predictor count.")
    if (any(splitQuant > 1 || splitQuant < 0))
        stop("Split specification contains invalid quantile values.")
  }
    
  if (nSamp == 0) {
    nSamp <- ifelse(withRepl, nRow, round((1-exp(-1)) * nRow))
  }
    
  if (maxLeaf < 0)
      stop("Leaf maximum must be nonnegative.")
  if (maxLeaf > nSamp)
      warning("Specified leaf maximum exceeds number of samples.")

  if (predProb != 0.0 && predFixed != 0)
      stop("Conflicting predictor sampling specifications:  Bernoulli and fixed.")
  if (predFixed == 0) {
    predFixed <- ifelse(predProb != 0.0, 0, ifelse(nPred >= 16, 0, ifelse(!is.factor(y), max(floor(nPred/3), 1), floor(sqrt(nPred)))))
  }
  if (predProb == 0.0) {
    predProb <- ifelse(predFixed != 0, 0.0, ifelse(!is.factor(y), 0.4, ceiling(sqrt(nPred))/nPred))
  }

  if (is.null(predWeight)) {
    predWeight <- rep(1.0, nPred)
  }

  if (any(is.na(y)))
    stop("NA not supported in response")
  if (!is.numeric(y) && !is.factor(y))
    stop("Expecting numeric or factor response")

  # Class weights
  if (is.factor(y)) {
    # Allows for gaps:
    ctgWidth <- max(as.integer(y))
    if (!is.null(classWeight)) {
      if (is.numeric(classWeight)) {
        if (length(classWeight) != ctgWidth)
          stop("class weights must conform to response cardinality")
        if (any(classWeight < 0))
          stop("class weights must be nonnegative")
        if (all(classWeight == 0.0)) {
          stop("class weights cannot all be zero")
        }
      }
      else if (classWeight == "balance") { # place-holder value
        classWeight <- rep(0.0, ctgWidth)
      }
      else {
        stop("Unrecognized class weights")
      }
    }
    else {
      classWeight <- rep(1.0, ctgWidth)
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
  if (!is.null(rowWeight)) {
    if (length(rowWeight) != nRow)
      stop("Sample weight length must match row count")
    
    if (any(rowWeight < 0))
      stop("Negative weights not permitted")
  }
  else {
    rowWeight = rep(1.0, nRow)
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

  # Normalizes vector of pointwise predictor probabilites.
  #if (predProb != 0 && predFixed != 0)
   # stop("Only one of 'predProb' and 'predFixed' may be specified")
  if (length(predProb) > 1)
    stop("'predProb' must have a scalar value")
  if (predProb < 0 || predProb > 1.0)
    stop("'predProb' value must lie in [0,1]")
  if (predFixed < 0 || predFixed > nPred)
    stop("'predFixed' must be positive integer <= predictor count")

  meanWeight <- ifelse(predProb == 0.0, 1.0, predProb)
  probVec <- predWeight * (nPred * meanWeight) / sum(predWeight)

  if (is.factor(y)) {
    if (any(regMono != 0)) {
      stop("Monotonicity undefined for categorical response")
    }
    train <- tryCatch(.Call("RcppTrainCtg", predBlock, preFormat$rowRank, y, nTree, nSamp, rowWeight, withRepl, treeBlock, minNode, minInfo, nLevel, maxLeaf, predFixed, splitQuant, probVec, autoCompress, thinLeaves, FALSE, classWeight), error=function(e){stop(e)})
  }
  else {
    train <- tryCatch(.Call("RcppTrainReg", predBlock, preFormat$rowRank, y, nTree, nSamp, rowWeight, withRepl, treeBlock, minNode, minInfo, nLevel, maxLeaf, predFixed, splitQuant, probVec, autoCompress, thinLeaves, FALSE, regMono), error=function(e) {stop(e)})
  }

  predInfo <- train[["predInfo"]]
  names(predInfo) <- predBlock$colnames
    training = list(
        call = match.call(),
        info = predInfo,
        version = (sessionInfo())$otherPkgs$Rborist$Version,
        diag = train[["diag"]]
  )

  if (noValidate) {
    validation <- NULL
  }
  else {
    validation <- Validate(preFormat, train, y, ctgCensus, quantVec, quantiles, qBin)
  }

  arbOut <- list(
    forest = train$forest,
    leaf = train$leaf,
    signature = predBlock$signature,
    training = training,
    validation = validation
  )
  class(arbOut) <- "Rborist"

  arbOut
}


# Groups predictors into like-typed blocks and creates zero-based type
# summaries.
#
PredBlock <- function(x, sigTrain = NULL) {
  # Argument checking:
  if (any(is.na(x)))
    stop("NA not supported in design matrix")

  # For now, only numeric and factor types supported.
  #
  if (is.data.frame(x)) { # As with "randomForest" package
    facCard <- as.integer(sapply(x, function(col) ifelse(is.factor(col) && !is.ordered(col), length(levels(col)), 0)))
    numCols <- as.integer(sapply(x, function(col) ifelse(is.numeric(col), 1, 0)))
    facIdx <- which(facCard > 0)
    numIdx <- which(numCols > 0)
    if (length(numIdx) + length(facIdx) != ncol(x)) {
      stop("Frame column with unsupported data type")
    }
    return(tryCatch(.Call("RcppPredBlockFrame", x, numIdx, facIdx, facCard, sigTrain), error = function(e) {stop(e)} ))
  }
  else if (is.matrix(x)) {
    if (is.integer(x)) {
      return(tryCatch(.Call("RcppPredBlockNum", data.matrix(x)), error=function(e) {stop(e)} ))
    }
    else if (is.numeric(x)) {
      return(tryCatch(.Call("RcppPredBlockNum", x), error=function(e) {stop(e)}))
    }
    else if (is.character(x)) {
      stop("Character data not yet supported")
    }
    else {
      stop("Unsupported matrix type")
    }
  }
  else if (inherits(x, "dgCMatrix")) {
     return(tryCatch(.Call("RcppPredBlockSparse", x), error= print))
  }
  else {
    stop("Expecting data frame or matrix")
  }
}


# Uses quartiles by default.
#
DefaultQuantVec <- function() {
  seq(0.25, 1.0, by = 0.25)
}
