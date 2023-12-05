# Copyright (C)  2012-2023   Mark Seligman
##
## This file is part of sgbArb.
##
## sgbArb is free software: you can redistribute it and/or modify it
## under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 2 of the License, or
## (at your option) any later version.
##
## sgbArb is distributed in the hope that it will be useful, but
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

sgbArb <- function(x, y, ...) UseMethod("sgbArb")


sgbArb.default <- function(x,
                             y,
                             autoCompress = 0.25,
                             impPermute = 0,
                             indexing = FALSE,
                             maxLeaf = 0,
                             minInfo = 0.01,
                             minNode = if (is.factor(y)) 2 else 3,
                             nLevel = 6,
                             nSamp = length(y) / 2,
                             nThread = 0,
                             nTree = 100,
                             withRepl = FALSE,
                             noValidate = TRUE,
                             nu = 0.1,
                             predFixed = ncol(x),
                             predProb = 0.0,
                             predWeight = NULL, 
                             regMono = NULL,
                             samplingWeight = NULL,
                             splitQuant = NULL,
                           stopLag = 0,
                             thinLeaves = is.factor(y) && !indexing,
                           trackFit = stopLag > 0,
                             trapUnobserved = FALSE,
                             treeBlock = 1,
                             verbose = FALSE,
                             ...) {

    # Argument checking:
    if (nThread < 0) {
        warning("Thread count must be nonnegative:  substituting default.")
        nThread <- 0
    }

    preFormat <- preformat(x, verbose)
    sampler <- presample(y, samplingWeight, nSamp, nTree, withRepl, verbose)
    train <- sgbTrain(preFormat, sampler, y,
                      autoCompress,
                      maxLeaf,
                      minInfo,
                      minNode,
                      nLevel,
                      nThread,
                      nTree,
                      nu,
                      predFixed,
                      predProb,
                      predWeight,
                      regMono,
                      splitQuant,
                      stopLag,
                      thinLeaves,
                      trackFit,
                      treeBlock,
                      verbose
                      )

    if (noValidate) {
        summaryValidate <- NULL
        if (impPermute > 0)
            warning("Variable importance requires validation:  ignoring")
    }
    else {
        argPredict <- list(
            bagging = FALSE,
            impPermute = impPermute,
            ctgProb = ctgProbabilities(sampler, "prob"),
            quantVec = NULL,
            indexing = indexing,
            trapUnobserved = trapUnobserved,
            nThread = nThread,
            verbose = verbose
            )
        # can validate without prediction if permutation tests not requested:
        # summaryValidate <- validate(train$sampler, train$leaf) 
        summaryValidate <- validateCommon(train, sampler, preFormat, argPredict)
    }
    postTrain(sampler, train, summaryValidate, impPermute)
}


postTrain <- function(sampler, train, summaryValidate, impPermute) {
    training = list(
        call = match.call(),
        version = train$version,
        diag = train$diag,
        samplerHash = train$samplerHash,
        signature = train$signature
    )

    # Consider caching train object and avoid copying its individual
    # members:
    if (impPermute > 0) {
        arbOut <- list(
            sampler = sampler,
            leaf = train$leaf,
            forest = train$forest,
            predMap = train$predMap,
            signature = train$signature,
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
            signature = train$signature,
            training = training,
            prediction = summaryValidate$prediction,
            validation = summaryValidate$validation
        )

    }
    class(arbOut) <- c("sgbArb", "sgbTrain")

    arbOut
}

