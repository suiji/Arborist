% File man/predict.Rborist.Rd
% Part of the rborist package

\name{predict.rfArb}
\alias{predict.rfArb}
\title{predict method for Rborst}
\description{
  Prediction and test using Rborist.
}

\usage{
\method{predict}{rfArb}(object, newdata, yTest=NULL,
keyedFrame = FALSE, quantVec=NULL, quantiles = !is.null(quantVec),
ctgCensus = "votes", indexing = FALSE, trapUnobserved = FALSE,
bagging = FALSE, nThread = 0, verbose = FALSE, ...)
}

\arguments{
  \item{object}{an object of class \code{rfArb}, created from a
    previous invocation of the command \code{Rborist} or \code{rfArb} to
    train.}
  \item{newdata}{a design frame or matrix containing new data, with the
    same signature of predictors as in the training command.}
  \item{yTest}{a response vector against which to test the new
    predictions.}
  \item{keyedFrame}{whether the columns of \code{newdata} may appear in
    arbitrary order or as a superset of the predictors used to train.}
  \item{quantVec}{a vector of quantiles to predict.}
  \item{quantiles}{whether to predict quantiles.}
  \item{ctgCensus}{whether/how to summarize per-category predictions.
  "votes" specifies the number of trees predicting a given class.
  "prob" specifies a normalized, probabilistic summary.
  "probSample" specifies sample-weighted probabilities, similar to
  quantile histogramming.}
  \item{indexing}{whether to record the final node index, typically
  terminal, of tree traversal.}
  \item{trapUnobserved}{reports score for nonterminal upon encountering
  values not observed during training, such as missing data.}
  \item{bagging}{whether prediction is restricted to out-of-bag samples.}
  \item{nThread}{suggests ans OpenMP-style thread count.  Zero denotes
    default processor setting.}
  \item{verbose}{whether to output progress of prediction.}
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
    \code{importance}{if permution importance requested, an object of
      class \code{importanceReg}, consisting of:
      \code{names}{the predictor names.}
      \code{mse}{the per-predictor mean-square error, under
	permutation.}
    }
  }
  \item{SummaryCtg}{an object of class \code{PredictCtg} consisting of:
      \code{yPred}{estimated categorical response.}
      \code{census}{factor-valued matrix of the estimate, by
	category, if requested.}
      \code{prob}{matrix of estimate probabilities, by category, if
	requested.}
      \code{indices}{final index of prediction, if requested.}
    \code{validation}{if validation requested, an object of class
      \code{ValidCtg} consisting of:
      \code{confusion}{the confusion matrix.}
      \code{misprediction}{the misprediction rate.}
      \code{oobError}{the out-of-bag error.}
    }
    \code{importance}{if permution importance requested, an object of
      class \code{importanceCtg}, consisting of:
      \code{mispred}{the misprediction rate, by predictor.}
      \code{oobErr}{the out-of-bag error, by predictor.}
    }
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
  
    