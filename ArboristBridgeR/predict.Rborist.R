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

"predict.Rborist" <- function(object, x, yTest=NULL, quantVec = NULL, ...) {
  if (!inherits(object, "Rborist"))
    stop("object not of class Rborist")
  if (is.null(object$forest))
    stop("Insufficient state maintained for prediction")
  PredictForest(object$forest, x, yTest, quantVec)
}

PredictForest <- function(forest, x, yTest, quantVec) {
  if (is.null(forest$predictors))
    stop("Unset predictors in forest")
  if (is.null(forest$splitValues))
    stop("Unset split values in forest")
  if (is.null(forest$scores))
    stop("Unset scores in forest")
  if (is.null(forest$bump))
    stop("Unset bump table in forest")

  quantiles <- !is.null(forest$quant)
  if (!quantiles && !is.null(quantVec))
    stop("Quantile vector given but no quantile state found")
  if (quantiles && is.null(quantVec))
    quantVec <- DefaultQuantVec()
  
  # Checks test data for conformity with training data.
  PredBlock(x)

  .Call("RcppReload", forest$predictors, forest$splitValues, forest$scores, forest$bump, forest$origins, forest$facOff, forest$facSplits)
  if (quantiles)
    .Call("RcppReloadQuant", forest$quant$qYRanked, forest$quant$qRankOrigin, forest$quant$qRank, forest$quant$qRankCount, forest$quant$qLeafPos, forest$quant$qLeafExtent)
  
  if (is.null(forest$ctgWidth)) {
    y <- numeric(nrow(x))
    if (!quantiles) {
      qPred <- NULL
      .Call("RcppPredictReg", y)
    }
    else {
      qPred <- numeric(nrow(x) *length(quantVec))
      .Call("RcppPredictQuant", quantVec, qPred, y)
      qPred <- matrix(qPred, nrow(x), length(quantVec), byrow=TRUE)
    }
    
    if (!is.null(yTest))
      val <- sum((y-yTest)^2) / length(y)
    else
      val <- y
    
    if (!is.null(qPred)) {
      if (!is.null(yTest)) {
        ret <- list(mse = val, quantiles = qPred)
      }
      else
        ret <- list(yPred = y, quantiles = qPred)
    }
    else {
      if (!is.null(yTest)) {
        ret <- list(mse = val)
      }
      else
        ret <- list(yPred = y)
    }
  }
  else {
    y <- integer(nrow(x))
    ctgWidth <- forest$ctgWidth
    unused <- .Call("RcppPredictCtg", y, ctgWidth)
    if (is.null(yTest)) {
      ret <- list(yPred = as.factor(y))
    }
    else {
      conf <- matrix(0L, ctgWidth, ctgWidth)
      for (i in 1:length(y)) {
        yActual <- yTest[i]
        yPred <- y[i]
        conf[yActual, yPred] <- conf[yActual, yPred] + 1
      }
      ret <- list(confusion = conf)
    }
  }

  ret
}
