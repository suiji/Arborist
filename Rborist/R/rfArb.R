# Copyright (C)  2012-2025   Mark Seligman
##
## This file is part of Rborist.
##
## Rborist is free software: you can redistribute it and/or modify it
## under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 2 of the License, or
## (at your option) any later version.
##
## Rborist is distributed in the hope that it will be useful, but
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
                classWeight = numeric(0),
                discardState = FALSE,
                impPermute = 0,
                indexing = FALSE,
                maxLeaf = 0,
                minInfo = 0.01,
                minNode = if (is.factor(y)) 2 else 3,
                nHoldout = 0,
                nLevel = 0,
                nSamp = 0,
                nThread = 0,
                nTree = 500,
                noValidate = FALSE,
                predFixed = 0,
                predProb = 0.0,
                predWeight = numeric(0),
                quantVec = numeric(0),
                quantiles = length(quantVec) > 0,
                regMono = numeric(0),
                rowWeight = numeric(0),
                samplingWeight = numeric(0),
                splitQuant = numeric(0),
                streamline = FALSE,
                thinLeaves = streamline || (is.factor(y) && !indexing),
                trapUnobserved = FALSE,
                treeBlock = 1,
                verbose = FALSE,
                withRepl = TRUE,
                ...) {
    if (nThread < 0) {
        warning("Thread count must be nonnegative:  substituting zero.")
        nThread <- 0
    }

    if (length(rowWeight) > 0) {
        warning("rowWeight will be deprecated.  Please use equivalent option 'samplingWeight'.")
        if (length(samplingWeight) > 0) {
            samplingWeight <- rowWeight
        }
    }
    
    # Quantile validation requires populated leaves.
    if (quantiles && thinLeaves) {
        warning("Quantile validation requested:  disabling thin leaves.")
        thinLeaves <- FALSE
    }
    
    preFormat <- preformat(x, verbose)
    sampler <- presample(y, samplingWeight, nSamp, nTree, withRepl, nHoldout, verbose=verbose)
    train <- rfTrain(preFormat, sampler, y,
                     autoCompress,
                     ctgCensus,
                     classWeight,
                     maxLeaf,
                     minInfo,
                     minNode,
                     nLevel,
                     nThread,
                     predFixed,
                     predProb,
                     predWeight,
                     regMono,
                     splitQuant,
                     thinLeaves,
                     treeBlock,
                     verbose)

    if (noValidate) {
        summaryValidate <- NULL
        if (impPermute > 0)
            warning("Permutation importance requires validation:  ignoring")
    }
    else {
        argPredict <- list(
            bagging = TRUE,
            impPermute = impPermute,
            ctgProb = ctgProbabilities(sampler, ctgCensus),
            quantVec = getQuantiles(quantiles, sampler, quantVec),
            indexing = indexing,
            trapUnobserved = trapUnobserved,
            nThread = nThread,
            verbose = verbose)
        # can validate without prediction if permutation tests not requested:
        # summaryValidate <- validate(train$sampler, train$leaf) 
        summaryValidate <- validateCommon(train, sampler, preFormat, argPredict)
    }

    postTrain(sampler, train, summaryValidate, impPermute, discardState)
}


postTrain <- function(sampler, train, summaryValidate, impPermute, discardState) {
    predInfo <- train$predInfo
    names(predInfo) <- train$signature$colNames
    training = list(
        call = match.call(),
        info = predInfo,
        version = train$version,
        diag = train$diag,
        samplerHash = train$samplerHash,
        signature = train$signature
    )

    # Consider caching train object ut avoid copying its individual
    # members:
    if (discardState) {
        if (impPermute > 0) {
            arbOut <- list(
                sampler = NULL,
                leaf = NULL,
                forest = NULL,
                signature = NULL,
                training = NULL,
                prediction = summaryValidate$prediction,
                validation = summaryValidate$validation,
                importance = summaryValidate$importance
            )
        }
        else {
            arbOut <- list(
                sampler = NULL,
                leaf = NULL,
                forest = NULL,
                signature = NULL,
                training = NULL,
                prediction = summaryValidate$prediction,
                validation = summaryValidate$validation
            )
        }
    }
    else {
        if (impPermute > 0) {
            arbOut <- list(
                sampler = sampler,
                leaf = train$leaf,
                forest = train$forest,
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
                signature = train$signature,
                training = training,
                prediction = summaryValidate$prediction,
                validation = summaryValidate$validation
            )
        }
    }

    structure(arbOut, class = "rfArb")
}

