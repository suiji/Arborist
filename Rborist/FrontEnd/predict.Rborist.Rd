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
quantiles = !is.null(quantVec), ctgCensus = "votes", oob = FALSE,
nThread = 0, verbose = FALSE, ...)
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
  \item{ctgCensus}{whether/how to summarize per-category predictions.
  "votes" specifies the number of trees predicting a given class.
  "prob" specifies a normalized, probabilistic summary.}
  \item{oob}{whether prediction is restricte to out-of-bag samples.}
  \item{nThread}{suggests ans OpenMP-style thread count.  Zero denotes
    default processor setting.}
  \item{verbose}{whether to output progress of prediction.}
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


\examples{
\dontrun{
  # Regression example:
  nRow <- 5000
  x <- data.frame(replicate(6, rnorm(nRow)))
  y <- with(x, X1^2 + sin(X2) + X3 * X4) # courtesy of S. Welling.
  rb <- Rborist(x,y)


  # Performs separate prediction on new data:
  xx <- data.frame(replace(6, rnorm(nRow)))
  pred <- predict(rb, xx)
  yPred <- pred$yPred


  # Performs separate prediction, using original response as test
  # vector:
  pred <- predict(rb, xx, y)
  mse <- pred$mse
  rsq <- pred$rsq


  # Performs separate prediction with (default) quantiles:
  pred <- predict(rb, xx, quantiles="TRUE")
  qPred <- pred$qPred


  # Performs separate prediction with deciles:
  pred <- predict(rb, xx, quantVec = seq(0.1, 1.0, by = 0.10))
  qPred <- pred$qPred


  # Classification examples:
  data(iris)
  rb <- Rborist(iris[-5], iris[5])


  # Generic prediction using training set.
  # Census as (default) votes:
  pred <- predict(rb, iris[-5])
  yPred <- pred$yPred
  census <- pred$census


  # As above, but validation census to report class probabilities:
  pred <- predict(rb, iris[-5], ctgCensus="prob")
  prob <- pred$prob


  # As above, but with training reponse as test vector:
  pred <- predict(rb, iris[-5], iris[5], ctgCensus = "prob")
  prob <- pred$prob
  conf <- pred$confusion
  misPred <- pred$misPred
}
}

\author{
  Mark Seligman at Suiji.
}

\seealso{\code{\link{Rborist}}}
  
    