# Copyright (C)  2012-2014   Mark Seligman
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
# TODO:  Replace minHeight default for factors to 1 from 3.
#
"Rborist.default" <- function(x, y, numTrees=500, smpWithRepl = TRUE,
                #importance = FALSE,
                #nPermute = ifelse(importance, 1, 0),
                minHeight = ifelse(!is.factor(y), 6, 2),
                predProb = ifelse(!is.factor(y), 0.4, sqrt(ncol(x))/ncol(x)),
                predWeight = rep(1.0, ncol(x)),
                nSamp = ifelse(smpWithRepl, nrow(x), round((1-exp(-1))*nrow(x))),
                sampWeight = NULL,
                quantVec = NULL,
                quantiles = !is.null(quantVec),
                pvtMinRatio = 0.01,
                pvtLevels = 0, pvtBlock = 8, pvtNoPredict = FALSE) {

  # Argument checking:
  if (any(is.na(x)))
    stop("NA not supported in design matrix")
  if (any(is.na(y)))
    stop("NA not supported in response")

  # Height constraints
  if (minHeight < 2)
    stop("Minimum splitting height must be at least 2")
  else if (minHeight > nSamp)
    stop("Minimum splitting height exceeds tree size")

  # Predictor weight constraints
  if (length(predWeight) != ncol(x))
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
  if (!is.null(sampWeight)) {
    if (length(sampWeight) != nrow(x))
      stop("Sample weight length must match row count")
    
    if (any(sampWeight < 0))
      stop("Negative weights not permitted")
  }
  
  # Quantile constraints:  regression only
  if (quantiles && is.factor(y))
    stop("Quantiles supported for regression case only")

  if (!is.null(quantVec)) {
    if (any(quantVec > 1) || any(quantVec < 0))
      stop("Quantile range must be within [0,1]")
    if (any(diff(quantVec) <= 0))
      stop("Quantile range must be increasing")
  }

   # L'Ecuyer's CMRG has more desirable distributional properties for this application.
  saveRNG <- RNGkind()[1]
  RNGkind("L'Ecuyer-CMRG")
  BlockData(x,y)
  .Call("RcppTrainInit", predWeight, predProb, numTrees, nSamp, smpWithRepl, quantiles, pvtMinRatio, pvtBlock);
  ctgWidth <- .Call("RcppTrainResponse", y)
  
  if (!is.null(sampWeight))
    unused <- .Call("RcppSampWeight", sampWeight)
  facWidth <- integer(1)
  totBagCount <- integer(1)
  totQLeafWidth <- integer(1)
  height <- .Call("RcppTrain", minHeight, facWidth, totBagCount, totQLeafWidth, pvtLevels)
  
  # The forest consists of trees specified by a splitting predictor and value, as
  # well as a Gini coefficient and subtree mean.
  predGini <- numeric(ncol(x))

  if (pvtNoPredict) {
    error <- -1
    confusion <- -1
  }
  else {
    unused <- BlockData(x)
    if (is.factor(y)) {
      error <- numeric(nlevels(y))
      confusion <- matrix(0L, nlevels(y), nlevels(y))
      unused <- .Call("RcppPredictOOBCtg", predGini, confusion, error)
    }
    else {
      error <- numeric(1)
      if (quantiles) {
        if (is.null(quantVec))
          quantVec <- DefaultQuantVec()
        qPred <- numeric(nrow(x) * length(quantVec))
        unused <- .Call("RcppPredictOOBQuant", predGini, error, quantVec, qPred)
        qPred <- matrix(qPred, nrow(x), length(quantVec), byrow = TRUE)
      }
      else {
        qPred <- NULL
        unused <- .Call("RcppPredictOOB", predGini, error)
      }
    }
  }
  
  if (quantiles) {
    qYRanked <- numeric(nrow(x))
    qRankOrigin <- integer(numTrees)
    qRank <- integer(totBagCount)
    qRankCount <- integer(totBagCount)
    qLeafPos <- integer(totQLeafWidth)
    qLeafExtent <- integer(totQLeafWidth)

    unused <- .Call("RcppWriteQuantile", qYRanked, qRankOrigin, qRank, qRankCount, qLeafPos, qLeafExtent)
    qOut <- list(qYRanked = qYRanked,
                  qRankOrigin = qRankOrigin,
                  qRank = qRank,
                  qRankCount = qRankCount,
                  qLeafPos = qLeafPos,
                  qLeafExtent = qLeafExtent
                  )
  }
  else
    qOut <- NULL

  preds <-  integer(height)
  splits <- numeric(height)
  leafScores <- numeric(height)
#  splitGini <- matrix(0.0, height, numTrees)
  facSplits <- integer(as.integer(facWidth))
  bumpL <- integer(height)
  bumpR <- integer(height)
  facOff <- integer(numTrees)
  origins <- integer(numTrees)
  unused <- .Call("RcppWriteForest", preds, splits, leafScores, bumpL, bumpR, origins, facOff, facSplits);


  RNGkind(saveRNG)
  if (is.factor(y)) {
    arbOut <- list(
                forest = list(
                  predictors = preds,
                  splitValues = splits,
                  scores = leafScores,
                  bumpL = bumpL,
                  bumpR = bumpR,
                  origins = origins,
                  facOff = facOff,
                  facSplits = facSplits,
                  ctgWidth = ctgWidth),
#                splitGini = splitGini,
                misprediction = error,
                Gini = predGini,
                confusion = confusion
                )
  }
  else {
    mse <- error[1]
    arbOut <- list(
                forest = list(
                  predictors = preds,
                  splitValues = splits,
                  scores = leafScores,
                  bumpL = bumpL,
                  bumpR = bumpR,
                  origins = origins,
                  facOff = facOff,
                  facSplits = facSplits,
                  quant = qOut),
#                splitGini = splitGini,
                mse = mse,
                rsq = 1 - (mse*nrow(x))/ (var(y) * (nrow(x) - 1)),
                qPred = qPred,
                Gini = predGini
                )
  }
  class(arbOut) <- "Rborist"
  arbOut
}

# Breaks data into blocks suitable for Rcpp methods.
#
BlockData <- function(x, y = NULL){#, quantiles = NULL) {
  training <- ifelse(is.null(y), FALSE, TRUE)

  # For now, turns off any special handling of Integer and Character to process as numeric:
  #
  if (is.data.frame(x)) { # As with "randomForest" package
    facLevels <- as.integer(sapply(x, function(col) ifelse(is.factor(col) && !is.ordered(col), length(levels(col)), 0)))
    numCols <- as.integer(sapply(x, function(col) ifelse(is.numeric(col), 1, 0)))
    nFacCol <- length(which(facLevels > 0))
    nNumCol <- length(which(numCols > 0))
    if (nFacCol + nNumCol != ncol(x))
      stop("Non-numeric, non-factor data appear among observations")
    
    .Call("RcppPredictorFrame", x, nrow(x), ncol(x), nFacCol, nNumCol, facLevels)
  }
  else if (is.integer(x)) {
    .Call("RcppPredictorNum", data.matrix(x), TRUE)
  }
  else if (is.numeric(x)) {
    if (training) {
      .Call("RcppPredictorNum", x, TRUE)
    }
    else {
      .Call("RcppPredictorNum", x, FALSE)
    }
  }
  else if (is.character(x)) {
    stop("Character data not yet supported");
  }
  else {
    stop("Unsupported data format");
  }
}

DefaultQuantVec <- function() {
  seq(0.25, 1.0, by = 0.25)
}
