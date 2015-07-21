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

"predict.Rborist" <- function(object, newdata, yTest=NULL, quantVec = NULL, qBin = 1000, ctgCensus = NULL, ...) {
  if (!inherits(object, "Rborist"))
    stop("object not of class Rborist")
  if (is.null(object$forest))
    stop("Insufficient state maintained for prediction")
  PredictForest(object$forest, newdata, yTest, quantVec, qBin, ctgCensus)
}

PredictForest <- function(forest, newdata, yTest, quantVec, qBin, ctgCensus) {
  if (is.null(forest$predictors))
    stop("Unset predictors in forest")
  if (is.null(forest$splitValues))
    stop("Unset split values in forest")
  if (is.null(forest$bump))
    stop("Unset bump table in forest")

  quantiles <- !is.null(forest$quant)
  if (!quantiles && !is.null(quantVec))
    stop("Quantile vector given but no quantile state found")
  if (quantiles && is.null(quantVec))
    quantVec <- DefaultQuantVec()
  
  # Checks test data for conformity with training data.
  PredBlock(newdata)

  .Call("RcppReload", forest$predictors, forest$splitValues, forest$bump, forest$origins, forest$facOff, forest$facSplits)
  if (quantiles)
    .Call("RcppReloadQuant", length(forest$origins), forest$quant$qYRanked, forest$quant$qRank, forest$quant$qSCount)
  
  if (is.null(forest$levels)) {
    y <- numeric(nrow(newdata))
    if (!quantiles) {
      qPred <- NULL
      .Call("RcppPredictReg", y)
    }
    else {
      qPred <- numeric(nrow(newdata) * length(quantVec))
      .Call("RcppPredictQuant", quantVec, qBin, qPred, y)
      qPred <- matrix(qPred, nrow(newdata), length(quantVec), byrow=TRUE)
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
        ret <- list(predicted = y, quantiles = qPred)
    }
    else {
      if (!is.null(yTest)) {
        ret <- list(mse = val)
      }
      else
        ret <- list(predicted = y)
    }
  }
  else {
    y <- integer(nrow(newdata))
    trainClass = forest$levels
    ctgWidth = length(trainClass)
    if (is.null(ctgCensus)) {
      census <- NULL
      prob <- FALSE
    }
    else if (ctgCensus == "votes") {
      census <- integer(nrow(newdata) * ctgWidth)
      prob <- FALSE
    }
    else if (ctgCensus == "prob") {
      census <- integer(nrow(newdata) * ctgWidth)
      prob <- TRUE
    }
    else {
      stop(paste("Unrecognized ctgCensus type:  ", ctgCensus))
    }


    unused <- .Call("RcppPredictCtg", y, ctgWidth, census)
    if (!is.null(census)) {
      census <- matrix(census, nrow(newdata), ctgWidth, byrow = TRUE)
      if (prob) {
        census <- t(apply(census, 1, function(x) { x / sum(x) }))
      }
      dimnames(census) <- list(rownames(newdata), trainClass)
    }

    if (is.null(yTest)) {
      ret <- list(predicted = as.factor(y), ctgCensus=census)
    }
    else {
      conf <- matrix(0L, ctgWidth, ctgWidth)
      for (i in 1:length(y)) {
        yActual <- yTest[i]
        yPred <- y[i]
        conf[yActual, yPred] <- conf[yActual, yPred] + 1
      }
      ret <- list(confusion = conf, ctgCensus=census)
    }
  }

  ret
}
