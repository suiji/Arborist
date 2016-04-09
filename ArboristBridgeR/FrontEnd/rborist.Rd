% File man/rborist.Rd
% Part of the rborist package

\name{Rborist}
\alias{Rborist}
\alias{Rborist.default}
\concept{decision trees}
\title{Rapid Decision Tree Construction and Evaluation}
\description{
  Accelerated implementation of the Random Forest (trademarked name)
  algorithm.  Tuned for multicore and GPU hardware.  Bindable with most
  numerical front-end languages in addtion to R.  Invocation is
  similar to that provided by "randomForest" package.
}

\usage{
\method{Rborist}{default} (x, y, nTree=500, withRepl = TRUE,
                ctgCensus = "votes",
                classWeight = NULL,
                minInfo = 0.01,
                minNode = ifelse(is.factor(y), 2, 5),
                nLevel = 0,
                noValidate = FALSE,
                nSamp = 0,
                predFixed = 0,
                predProb = 0.0,
                predWeight = NULL, 
                qVec = NULL,
                quantiles = !is.null(qVec),
                qBin = 5000,
                regMono = NULL,
                rowWeight = NULL,
                treeBlock = 1,
                pvtBlock = 8, ...)
}

\arguments{
  \item{x}{the design matrix expressed as a \code{PreTrain} object, as a
  \code{data.frame} object with numeric and/or \code{factor} columns or
  as a numeric matrix.}
  \item{y}{the response (outcome) vector, either numerical or
  categorical.  Row count must conform with \code{x}.}
  \item{nTree}{the number of trees to train.}
  \item{withRepl}{whether row sampling is by replacement.}
  \item{ctgCensus}{report categorical validation by vote or by probability.}
  \item{classWeight}{proportional weighting of classification categories.}
  \item{minInfo}{information ratio with parent below which node does not split.}
  \item{minNode}{minimum number of distinct row references to split a node.}
  \item{nLevel}{maximum number of tree levels to train.  Zero denotes no
    limit.}
  \item{noValidate}{whether to train without validation.}
  \item{nSamp}{number of rows to sample, per tree.}
  \item{predFixed}{number of trial predictors for a split (\code{mtry}).}
  \item{predProb}{probability of selecting individual predictor as trial splitter.}
  \item{predWeight}{relative weighting of individual predictors as trial
    splitters.}
  \item{qVec}{quantile levels to validate.}
  \item{quantiles}{whether to report quantiles at validation.}
  \item{qBin}{bin size for facilating quantiles at large sample count.}
  \item{regMono}{signed probability constraint for monotonic
    regression.}
  \item{rowWeight}{row weighting for initial sampling of tree.}
  \item{treeBlock}{maximum number of trees to train during a single
    level (e.g., coprocessor computing).}
  \item{pvtBlock}{maximum number of trees to train in a block (e.g.,
  cluster computing).}
  \item{...}{not currently used.}
}

\value{An object of class \code{Rborist}.  Contains a \code{forest}
  object, for subsequent prediction, as well as measure of prediction
  quality on the out-of-bag samples.  For regression, a mean-square
  error rate and R-squared values are reported in \code{mse} and
  \code{rsq}, respectively.  For categorical response, the misprediction
  rate and confusion matrices are reported \code{misprediction} and
  \code{confusion}, respectively.

  If quantile training has been requested, a \code{quantile} object, for
  subsequent quantile regression.
}


\author{
  Mark Seligman at Suiji.
}
