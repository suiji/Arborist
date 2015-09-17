# Copyright (C)  2012-20154   Mark Seligman
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
                ctgCensus = "votes",
                classWeight = NULL,
                nSamp = ifelse(withRepl, nrow(x), round((1-exp(-1))*nrow(x))),
                minInfo = 0.05,
                minNode = ifelse(is.factor(y), 2, 5),
                nLevel = 0,
                predFixed = 0,
                predProb = ifelse(predFixed != 0, 0, ifelse(!is.factor(y), 0.4, ceiling(sqrt(ncol(x)))/ncol(x))),
                predWeight = rep(1.0, ncol(x)),
                qVec = NULL,
                quantiles = !is.null(qVec),
                qBin = 5000,
                sampleWeight = NULL,
                treeBlock = 1,
                pvtBlock = 8, pvtNoPredict = FALSE, ...) {

  # Argument checking:
  if (any(is.na(x)))
    stop("NA not supported in design matrix")
  if (any(is.na(y)))
    stop("NA not supported in response")
  if (!is.numeric(y) && !is.factor(y))
    stop("Exping numeric or factor response")

  # Class weights
  if (!is.null(classWeight)) {
    if (is.numeric(classWeight)) {
    }
    else if (classWeight == "auto") {
    }
    else {
      stop("Unrecognized class weights")
    }
  }  
  
  # Height constraints
  if (minNode < 1)
    stop("Minimum node size must be positive")
  else if (minNode > nSamp)
    stop("Minimum splitting width exceeds tree size")

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
  if (!is.null(sampleWeight)) {
    if (length(sampleWeight) != nrow(x))
      stop("Sample weight length must match row count")
    
    if (any(sampleWeight < 0))
      stop("Negative weights not permitted")
  }
  else {
    sampleWeight = rep(1.0, nrow(x))
  }
  
  # Quantile constraints:  regression only
  if (quantiles && is.factor(y))
    stop("Quantiles supported for regression case only")

  if (!is.null(qVec)) {
    if (any(qVec > 1) || any(qVec < 0))
      stop("Quantile range must be within [0,1]")
    if (any(diff(qVec) <= 0))
      stop("Quantile range must be increasing")
  }

  if (pvtNoPredict)
    stop("Non-validating version NYI")
  
   # L'Ecuyer's CMRG has more desirable distributional properties for this application.
  saveRNG <- RNGkind()[1]
  RNGkind("L'Ecuyer-CMRG")

  # Normalizes vector of pointwise predictor probabilites.
  if (predProb != 0 && predFixed != 0)
    stop("Only one of 'predProb' and 'predFixed' may be specified")
  if (predProb < 0 || predProb > 1.0)
    stop("'predProb' value must lie in [0,1]");
  if (predFixed < 0 || predFixed > ncol(x))
    stop("'predFixed' must be positive integer <= predictor count")

  meanWeight <- ifelse(predProb == 0, 1.0, predProb)
  probVec <- predWeight * ((ncol(x) * meanWeight) / sum(predWeight))

  # Predictor and Sample immutables must be set by commencement of training.
  PredBlock(x, y, probVec, predFixed)
  unused <- .Call("RcppSample", nrow(x), nSamp, sampleWeight, withRepl)
  if (is.factor(y)) {
    train <- .Call("RcppTrainCtg", y, nTree, ncol(x), nSamp, treeBlock, minNode, minInfo, nLevel)
    forest = list(
       pred = train[["pred"]],
       split = train[["split"]],
       bump = train[["bump"]],
       origin = train[["origin"]],
       facOrig = train[["facOrig"]],
       facSplit = train[["facSplit"]],
       yLevels = levels(y)
    )
    
    leaf <- list(
      weight = train[["weight"]]
    )
  }
  else {
    train <- .Call("RcppTrainReg", y, nTree, ncol(x), nSamp, treeBlock, minNode, minInfo, nLevel)
    forest = list(
      pred = train[["pred"]],
      split = train[["split"]],
      bump = train[["bump"]],
      origin = train[["origin"]],
      facOrig = train[["facOrig"]],
      facSplit = train[["facSplit"]]
    )

    leaf <- list(
      yRanked = train[["yRanked"]],
      rank = train[["rank"]],
      sCount = train[["sCount"]]
    )
  }
  RNGkind(saveRNG)

  predInfo <- train[["predInfo"]]
  names(predInfo) <- colnames(x)
  training = list(
    info = predInfo,
    bag = train[["bag"]]
  )

  if (!pvtNoPredict) {
    PredBlock(x) # Cheap way to undo internal sort.
    if (is.factor(y)) {
      ctgWidth <- length(levels(y))
      unused <- .Call("RcppForestCtg", forest, leaf);
      error <- numeric(ctgWidth)
      conf <- rep(0L, ctgWidth * ctgWidth)
      yValid <- integer(length(y))
      census <- rep(0L, length(y) * ctgWidth)
      if (ctgCensus == "votes") {
        prob <- NULL
        unused <- .Call("RcppValidateVotes", y, training[["bag"]], yValid, conf, error, census)
      }
      else if (ctgCensus == "prob") {
        prob <- rep(0.0, length(y) * ctgWidth)
        unused <- .Call("RcppValidateProb", y, training[["bag"]], yValid, conf, error, census, prob)
        prob <- matrix(prob, length(y), ctgWidth, byrow = TRUE)
        dimnames(prob) <- list(rownames(x), levels(y))
      }
      else {
        stop(paste("Unrecognized ctgCensus type:  ", ctgCensus))
      }
      confusion <- matrix(conf, ctgWidth, ctgWidth, byrow = TRUE)
      dimnames(confusion) <- list(levels(y), levels(y))
      census <- matrix(census, nrow(x), ctgWidth, byrow=TRUE)
      dimnames(census) = list(rownames(x), levels(y))
      validation = list(
        yValid = levels(y)[yValid],
        misprediction = error,
        confusion = confusion,
        census = census,
        prob = prob
        )
    }
    else {
      unused <- .Call("RcppForestReg", forest, leaf)
      yValid <- numeric(nrow(x))
      if (quantiles) {
        if (is.null(qVec)) {
          qVec <- DefaultQuantVec()
        }
        qValid <- numeric(nrow(x) * length(qVec))
        unused <- .Call("RcppValidateQuant", qVec, qBin, qValid, yValid, training[["bag"]])
        qValid <- matrix(qValid, nrow(x), length(qVec), byrow = TRUE)
      }
      else {
        qValid <- NULL
        unused <- .Call("RcppValidateReg", yValid, training[["bag"]])
      }

      mse <- .Call("RcppMSE", yValid, y)
      validation = list(
        yValid = yValid,
        mse = mse,
        rsq = 1 - (mse * nrow(x)) / (var(y) * (nrow(x) - 1)),
        qValid = qValid
      )
    }
  }
  else {
    validation <- NULL
  }

  arbOut <- list(
    forest = forest,
    leaf = leaf,
    training = training,
    validation = validation
  )
  class(arbOut) <- "Rborist"

  arbOut
}


# Breaks data into blocks suitable for Rcpp methods.
#
PredBlock <- function(x, y = NULL, probVec = NULL, predFixed = 0) {
  training <- ifelse(is.null(y), FALSE, TRUE)
  
  # For now, only numeric and factor types supporte.
  #
  unused <- .Call("RcppPredictorFactory", probVec, predFixed, ncol(x), nrow(x))
  
  if (is.data.frame(x)) { # As with "randomForest" package
    facLevels <- as.integer(sapply(x, function(col) ifelse(is.factor(col) && !is.ordered(col), length(levels(col)), 0)))
    numCols <- as.integer(sapply(x, function(col) ifelse(is.numeric(col), 1, 0)))
    nFacCol <- length(which(facLevels > 0))
    nNumCol <- length(which(numCols > 0))
    unused <- .Call("RcppPredictorFrame", x, nNumCol, nFacCol, facLevels)
  }
  else if (is.integer(x)) {
    unused <- .Call("RcppPredictorNum", data.matrix(x), TRUE)
  }
  else if (is.numeric(x)) {
    if (training) {
      unused <- .Call("RcppPredictorNum", x, TRUE)
    }
    else {
      unused <- .Call("RcppPredictorNum", x, FALSE)
    }
  }
  else if (is.character(x)) {
    stop("Character data not yet supported");
  }
  else {
    stop("Unsupported data format");
  }

  blockStatus <- .Call("RcppPredictorBlockEnd");
  if (blockStatus != 0)
    stop("Unsupported data types appear among observations")
}


# Uses quartiles by default.
#
DefaultQuantVec <- function() {
  seq(0.25, 1.0, by = 0.25)
}
