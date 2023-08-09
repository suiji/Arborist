% File man/validate.Rd
% Part of the Rborist package

\name{validate}
\alias{validate}
\alias{validate.default}
\concept{decision tree validation}
\title{Separate Validation of Trained Decision Forest}
\description{
  Permits trained decision forest to be validated separately from training.
}

\usage{
\method{validate}{default}(train, sampler, preFormat = NULL,  ctgCensus
= "votes", impPermute = 0, quantVec = NULL, quantiles =
!is.null(quantVec), indexing = FALSE, trapUnobserved = FALSE, nThread = 0, verbose =
FALSE, ...)
}

\arguments{
  \item{train}{an object of class \code{Rborist} obtained from previous
    training.}
  \item{sampler}{summarizes the response and its per-tree samplgin.}
  \item{preFormat}{internal representation of the design matrix, of
    class \code{PreFormat}}
  \item{ctgCensus}{report categorical validation by vote or by probability.}
  \item{impPermute}{specifies the number of importance permutations:  0
  or 1.}
  \item{quantVec}{quantile levels to validate.}
  \item{quantiles}{whether to report quantiles at validation.}
  \item{indexing}{whether to report final index, typically terminal, of
  tree traversal.}
  \item{trapUnobserved}{indicates whether to return a nonterminal for
  values unobserved during training, such as missing data.}
  \item{nThread}{suggests an OpenMP-style thread count.  Zero denotes
    the default processor setting.}
  \item{verbose}{indicates whether to output progress of validation.}
  \item{...}{not currently used.}
}

\value{an object of one of two classes:
  \item{SummaryReg}{summarizing regression, consisting of:
    \code{prediction}{an object of class \code{PredictReg} consisting
      of:
      \code{yPred}{the estimated numerical response.}
      \code{qPred}{quantiles of prediction, if requested.}
      \code{qEst}{quantile of the estimate, if quantiles requested.}
      \code{indices}{final index of prediction, if requested.}
    }
    \code{validation}{if validation requested, an object of class
      \code{ValidReg} consisting of:
      \code{mse}{the mean-square error of the estimate.}
      \code{rsq}{the r-squared statistic of the estimate.}
      \code{mae}{the mean absolute error of the estimate.}
    }
  }
  \item{SummaryCtg}{summarizing classification, consisting of:
    \code{prediciton}{an object of class \code{PredictCtg} consisting
      of:
      \code{yPred}{estimated categorical response.}
      \code{census}{factor-valued matrix of the estimate, by
	category, if requested.}
      \code{prob}{matrix of estimate probabilities, by category, if
	requested.}
      \code{indices}{final index of prediction, if requested.}
    }
    \code{validation}{if validation requested, an object of class
      \code{ValidCtg} consisting of:
      \code{confusion}{the confusion matrix.}
      \code{misprediction}{the misprediction rate.}
      \code{oobError}{the out-of-bag error.}
    }
  }
}

\examples{
\dontrun{
    ## Trains without validation.
    rb <- Rborist(x, y, novalidate=TRUE)
    ...
    ## Delayed validation using a preformatted object.
    pf <- preformat(x)
    v <- validate(pf, rb, y)
  }
}


\author{
  Mark Seligman at Suiji.
}
