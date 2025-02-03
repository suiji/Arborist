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
\method{validate}{default}(train, preFormat, sampler = NULL,  ctgCensus
= "votes", impPermute = 0, quantVec = numeric(0), quantiles =
length(quantVec) > 0, indexing = FALSE, trapUnobserved = FALSE, nThread = 0, verbose =
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

\value{either of two pairs of objects:
  \itemize{
    \item \code{SummaryReg} summarizing regression, as documented with the
      command \code{predict.arbTrain}.
    \item \code{validation} an object of class \code{ValidReg} consisting of:
    \itemize{
      \item \code{mse} the mean-square error of the estimate.
      \item \code{rsq} the r-squared statistic of the estimate.
      \item \code{mae} the mean absolute error of the estimate.
    }
  }
  \itemize{
    \item \code{SummaryCtg} summarizing classification, as documented with the
      command \code{predict.arbTrain}.
    \item \code{validation} an object of class \code{ValidCtg} consisting of:
      \itemize{
	\item \code{confusion} the confusion matrix.
	\item \code{misprediction} the misprediction rate.
	\item \code{oobError} the out-of-bag error.
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
    v <- validate(rb, pf)
  }
}


\author{
  Mark Seligman at Suiji.
}
