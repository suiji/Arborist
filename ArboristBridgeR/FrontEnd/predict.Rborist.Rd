% File man/predict.rborist.Rd
% Part of the rborist package

\name{predict.Rborist}
\alias{predict.Rborist}
\title{predict method for Rborst}
\description{
  Prediction and test using Rborist.
}

\usage{
\method{predict}{Rborist}(object, newdata, yTest=NULL, quantVec=NULL,
quantiles = !is.null(quantVec), qBin = 5000, ctgCensus = "votes", ...)
}

\arguments{
  \item{object}{an object of class \code{Rborist}, created from a
    previous invocation of the command \code{Rborist} to train.}
  \item{newdata}{a design matrix containing new data, with the same signature
    of predictors as in the training command.}
  \item{yTest}{if specfied, a response vector against which to test the new
    predictions.}
  \item{quantVec}{a vector of quantiles to predict.}
  \item{quantiles}{whether to predict quantiles.}
  \item{qBin}{bin size for quantile etimation.  Performance scales with
    bin size.  Smaller bins sacrifice precision.}
  \item{ctgCensus}{whether/how to summarize per-category predictions.
  "votes" specifies the number of trees predicting a given class.
  "prob" specifies a normalized, probabilistic summary.}
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
  Mark Seligman at Suiji.
}

\seealso{\code{\link{Rborist}}}
  
    