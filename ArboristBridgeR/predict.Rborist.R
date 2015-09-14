# Copyright (C)  2012-2015   Mark Seligman
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

"predict.Rborist" <- function(object, newdata, yTest=NULL, qVec = NULL, quantiles = !is.null(qVec), qBin = 5000, ctgCensus = "votes", ...) {
  if (!inherits(object, "Rborist"))
    stop("object not of class Rborist")
  if (is.null(object$forest))
    stop("Forest state needed for prediction")
  if (quantiles && is.null(object$leaf))
    stop("Leaf state needed for quantile")
  if (quantiles && is.null(qVec))
    qVec <- DefaultQuantVec()

  PredictForest(object$forest, object$leaf, newdata, yTest, qVec, qBin, ctgCensus)
}

PredictForest <- function(forest, leaf, newdata, yTest, qVec, qBin, ctgCensus) {
  if (is.null(forest$pred))
    stop("Unset predictors in forest")
  if (is.null(forest$split))
    stop("Unset split values in forest")
  if (is.null(forest$bump))
    stop("Unset bump table in forest")

  if (!is.null(qVec)) {
    if (any(qVec > 1) || any(qVec < 0))
      stop("Quantile range must be within [0,1]")
    if (any(diff(qVec) <= 0))
      stop("Quantile range must be increasing")
  }

  # Checks test data for conformity with training data.
  PredBlock(newdata)

  nRow <- nrow(newdata)
  if (is.null(forest$yLevels)) {
    .Call("RcppForestReg", forest, leaf)
  
    yPred <- numeric(nRow)
    if (is.null(qVec)) {
      qPred <- NULL
      .Call("RcppPredictReg", yPred)
    }
    else {
      qPred <- numeric(nRow * length(qVec))
      .Call("RcppPredictQuant", qVec, qBin, qPred, yPred)
      qPred <- matrix(qPred, nRow, length(qVec), byrow=TRUE)
    }
    
    if (!is.null(yTest))
      mse <- .Call("RcppMSE", yPred, yTest)
    
    if (!is.null(qPred)) {
      if (!is.null(yTest)) {
        ret <- list(predicted = yPred, mse = mse, quantiles = qPred)
      }
      else
        ret <- list(predicted = yPred, quantiles = qPred)
    }
    else {
      if (!is.null(yTest)) {
        ret <- list(predicted = yPred, mse = mse)
      }
      else
        ret <- list(predicted = yPred)
    }
  }
  else {
    if (!is.null(qVec))
      stop("Quantiles supported for regression case only")

    unused <- .Call("RcppForestCtg", forest, leaf)
    yLevels <- forest$yLevels
    yPred <- integer(nRow)
    ctgWidth = length(yLevels)
    census <- rep(0L, nRow * ctgWidth)
    if (ctgCensus == "votes") {
      prob <- NULL
      unused <- .Call("RcppPredictVotes", yPred, census)
    }
    else if (ctgCensus == "prob") {
      prob <- rep(0.0, nRow * ctgWidth)
      unused <- .Call("RcppPredictProb", yPred, census, prob)
      prob <- matrix(prob, nRow, ctgWidth, byrow = TRUE)
      dimnames(prob) <- list(rownames(newdata), yLevels)
    }
    else {
      stop(paste("Unrecognized ctgCensus type:  ", ctgCensus))
    }
    census <- matrix(census, nRow, ctgWidth, byrow = TRUE)
    dimnames(census) <- list(rownames(newdata), yLevels)

    if (is.null(yTest)) {
      ret <- list(predicted = yLevels[yPred], census=census, prob=prob)
    }
    else {
      conf <- matrix(0L, ctgWidth, ctgWidth)
      for (i in 1:length(yPred)) {
        yActual <- yTest[i]
        yPred <- yPred[i]
        conf[yActual, yPred] <- conf[yActual, yPred] + 1
      }
      ret <- list(confusion = conf, census = census, prob=prob)
    }
  }

  ret
}
