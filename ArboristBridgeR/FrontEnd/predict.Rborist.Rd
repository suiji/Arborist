% File man/predict.Rborist.Rd
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

\value{ a list containing either of the two prediction containers:
    
  \item{PredictReg}{ a list of prediction results for regression:
      
  \code{yPred}{ a vector containing the predicted response.}

  \code{qPred}{ a matrix containing the prediction quantiles, if requested.}
  }

  \item{PredictCtg}{ a list of validation results for classification:
      
    \code{yPred}{ a vector containing the predicted response.}

    \code{census}{ a matrix of predictions, by category.}
    
    \code{prob}{ a matrix of prediction probabilities by category, if requested.}
  }
}

\author{
  Mark Seligman at Suiji.
}

\seealso{\code{\link{Rborist}}}
  
    