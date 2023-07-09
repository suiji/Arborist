% File man/predict.rfTrain.Rd
% Part of the Rborist package

\name{predict.rfTrain}
\alias{predict.rfTrain}
\title{predict method for rfTrain result}
\description{
  Prediction and test using Rborist.
}

\usage{
\method{predict}{rfTrain}(objTrain, newdata, sampler, yTest=NULL,
keyedFrame = FALSE, quantVec=NULL, quantiles = !is.null(quantVec),
ctgCensus = "votes", indexing = FALSE, trapUnobserved = FALSE,
bagging = FALSE, nThread = 0, verbose = FALSE, ...)
}

\arguments{
  \item{objTrain}{an object of class \code{rfTrain}, created from a
    previous invocation of the command \code{rfTrain} to
    train.}
  \item{newdata}{a design frame or matrix containing new data, with the
    same signature of predictors as in the training command.}
  \item{sampler}{an object of class \code{Sampler} used in the
  command.}
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

  pf <- preformat(x)
  sp <- presample(y)
  rb <- rfTrain(pf, sp, y)


  # Performs separate prediction on new data:
  xx <- data.frame(replace(6, rnorm(nRow)))
  pred <- predict(rb, xx)
  yPred <- pred$yPred

  # Additional examples appear in the documentation for
  # predict.rfArb.Rd, and can be applied mutatis mutandis.
}
}

\author{
  Mark Seligman at Suiji.
}

\seealso{\code{\link{rfTrain}}}
  
    