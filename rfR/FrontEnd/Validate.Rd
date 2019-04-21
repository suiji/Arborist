% File man/Validate.Rd
% Part of the Rborist package

\name{Validate}
\alias{Validate}
\alias{Validate.default}
\concept{decision tree validation}
\title{Separate Validation of Trained Decision Forest}
\description{
  Permits trained decision forest to be validated separately from training.
}

\usage{
\method{Validate}{default}(preFormat, train, y, ctgCensus = "votes",
quantVec = NULL, quantiles = !is.null(quantVec), nThread = 0, verbose = FALSE)
}

\arguments{
  \item{preFormat}{internal representation of the design matrix, of
    class \code{PreFormat}}
  \item{train}{an object of class \code{Rborist} obtained from previous
    training.}
  \item{y}{the response (outcome) vector, either numerical or
  categorical.  Row count must conform with \code{x}.}
  \item{ctgCensus}{report categorical validation by vote or by probability.}
  \item{quantVec}{quantile levels to validate.}
  \item{quantiles}{whether to report quantiles at validation.}
  \item{nThread}{suggests an OpenMP-style thread count.  Zero denotes
    the default processor setting.}
  \item{verbose}{indicates whether to output progress of validation.}
}

\value{an object of class \code{validation}:
  \item{validation}{ list containing either a:
    
    \code{ValidReg}{ list of validation results for regression:

      \code{yPred}{ vector containing the predicted response.}

      \code{mae}{ the mean absolute error of prediction.}
    
      \code{mse}{ the mean-square error of prediction.}

      \code{rsq}{ the r-squared statistic.}

      \code{qPred}{ matrix containing the prediction quantiles, if requested.}
  }
  or a:
  \code{ValidCtg}{ list of validation results for classification:
      
    \code{yPred}{ vector containing the predicted response.}

    \code{misprediction}{ vector containing the classwise misprediction rates.}
      
    \code{confusion}{ the confusion matrix.}

    \code{census}{ matrix of predictions, by category.}

    \code{oobError}{ the out-of-bag error.}

    \code{prob}{ matrix of prediction probabilities by category, if requested.}
  }
}
}
  


\examples{
\dontrun{
    ## Trains without validation.
    rb <- Rborist(x, y, noValidate=TRUE)
    ...
    ## Delayed validation using a PreFormat object.
    pf <- PreFormat(x)
    v <- Validate(pf, rb, y)
  }
}


\author{
  Mark Seligman at Suiji.
}
