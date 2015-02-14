% File man/predict.rborist.Rd
% Part of the rborist package

\name{predict.Rborist}
\alias{predict.Rborist}
\title{predict method for Rborst}
\description{
  Prediction and test using Rborist.
}

\usage{
\method{predict}{Rborist}(object,x,yTest=NULL, quantVec=NULL, ...)
}

\arguments{
  \item{object}{an object of class \code{Rborist}, created from a
    previous invocation of the command \code{Rborist} to train.}
  \item{x}{a design matrix containing new data, with the same signature
    of predictors as in the training command.}
  \item{yTest}{if specfied, a response vector against which to test the new
    predictions.}
  \item{quantVec}{a vector of quantiles from which to report new
  quantile regressions.  Only valid if training step specficied quantile
  regression.}
  \item{...}{not currently used.}
}

\value{
  A list with the following components:

  If regression, either a vector \code{yPred} containing the predictions for each
  row, if \code{yTest} is null, or the mean-square error, \code{mse} of the
  predictions computed with respect to expected \code{yTest}.

  If categorical, either a vector of predictions \code{yPred} for null
  \code{yTest} or a confusion matrix, \code{confusion}, computed with
  repsect to expected \code{yTest}.

  If training included quantile regression, the matrix \code{quantiles}
  records the vector of predicted quantiles for each row.
}

\author{
  Mark Seligman at Rapidics LLC.
}

\seealso{\code{\link{Rborist}}}
  
    