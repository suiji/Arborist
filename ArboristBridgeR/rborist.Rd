% File man/rborist.Rd
% Part of the rborist package

\name{Rborist}
\alias{Rborist}
\alias{Rborist.default}
\concept{decision trees}
\title{Rapid Decision Tree Construction and Evaluation}
\description{
  Accelerated implementation of the Random Forest (trademarked name)
  algorithm.  Tuned for multicore and GPU hardware.  Works with most
  numerical front-end languages in addtion to R.  Invocation is
  similar to that provided by "randomForest" package.
}

\usage{
\method{Rborist}{default}(x, y, nTree=500, withRepl = TRUE,
                predProb = ifelse(!is.factor(y), 0.4, sqrt(ncol(x))/ncol(x)),
                predWeight = rep(1.0, ncol(x)),
                nSamp = ifelse(withRepl, nrow(x), round((1-exp(-1))*nrow(x))),
                minNode = ifelse(!is.factor(y), 6, 2),
                nLevel = 0,
                minInfo = 0.01,
                sampleWeight = NULL,
                quantVec = NULL,
                quantiles = !is.null(quantVec),
                pvtBlock = 8,
                pvtNoPredict = FALSE, ...)
}

\arguments{
  \item{x}{the design matrix. Can be either a numerical matrix or a data
  frame of numerical or categorical columns.}
  \item{y}{the response vector.  Can be either numerical or categorical.
  Hence \code{x} and \code{y} should have the same number of rows.}
  \item{nTree}{the number of trees on which to train.}
  \item{withRepl}{whether row sampling is by replacement.}
  \item{predProb}{the probability that an arbitrary predictor is chosen
    as a splitting candidate.  Similar in spirit to the \code{mtry} option
    of \code{randomForest}, which is a fixed number of predictors over
    which to attempt to split at each node.}
  \item{predWeight}{a numerical vector of weights for selecting individual
    predictors as splitting candidates.}
  \item{nSamp}{the number of samples to draw when training an individual
    tree.}
  \item{minNode}{the minimal index set width over which to attempt
    splitting. Similar to \code{nodesize} parameter of
    \code{randomForest}.}
  \item{nLevel}{the number of levels at which to terminate.  At least
  one level of the decision tree is built.  The value \code{0} is reserved to
  stipulate no pre-set limit.}
  \item{minInfo}{a threshold ratio of information values between a
  subnode and its parent, below which splitting will not be attempted.}  
  \item{sampleWeight}{a numerical vector of weights for drawing individual
    rows as sample candidates.}
  \item{quantVec}{vector of quantiles over which to report quantile
    regression.}
  \item{quantiles}{whether to train with quantile information.  If
    \code{quantVec} is not specified but \code{quantiles} has been
    specified, then quartiles are employed by default.}
  \item{pvtBlock}{blocking parameter for performance tuning.  Currently
    unsupported.}
  \item{pvtNoPredict}{skips prediction.  Meant for testing only.}
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
