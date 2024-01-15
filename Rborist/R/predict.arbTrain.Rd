% File man/predict.rfTrain.Rd
% Part of the Rborist package

\name{predict.arbTrain}
\alias{predict.arbTrain}
\title{predict method for arbTrain result}
\description{
  Prediction and test using Rborist.
}

\usage{
\method{predict}{arbTrain}(object, newdata, sampler, yTest=NULL,
keyedFrame = FALSE, quantVec=NULL, quantiles = !is.null(quantVec),
ctgCensus = "votes", indexing = FALSE, trapUnobserved = FALSE,
bagging = FALSE, nThread = 0, verbose = FALSE, ...)
}

\arguments{
  \item{object}{an object of class \code{arbTrain}, created from a
    previous invocation of the command \code{rfArb}, \code{Rborist}
    or \code{rfTrain} to train.}
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
  \itemize{
    \item \code{SummaryReg} summarizing regression, consisting of:
    \itemize{
      \item \code{prediction} an object of class \code{PredictReg} consisting of:
      \itemize{
	\item \code{yPred} the estimated numerical response.
	\item \code{qPred} quantiles of prediction, if requested.
	\item \code{qEst} quantile of the estimate, if quantiles requested.
	\item \code{indices} final index of prediction, if requested.
      }
      \item \code{validation} if validation requested, an object of class \code{ValidReg} consisting of:
      \itemize{
	\item \code{mse} the mean-squared error of the estimate.
	\item \code{rsq} the r-squared statistic of the estimate.
	\item \code{mae} the mean absolute error of the estimate.
      }
      \item \code{importance} if permution importance requested, an object of class \code{importanceReg}, containing multiple instances of:
      \itemize{
	\item \code{names} the predictor names.
	\item \code{mse} the per-predictor mean-squared error, under permutation.
      }
    }
    \item \code{SummaryCtg} summarizing classification, consisting of:
    \itemize{
      \item \code{PredictCtg} consisting of:
      \itemize{
	\item \code{yPred} estimated categorical response.
	\item \code{census} factor-valued matrix of the estimate, by category, if requested.
	\item \code{prob} matrix of estimate probabilities, by category, if requested.
	\item \code{indices} final index of prediction, if requested.
      }
      \item \code{validation} if validation requested, an object of class \code{ValidCtg} consisting of:
      \itemize{
	\item \code{confusion} the confusion matrix.
	\item \code{misprediction} the misprediction rate.
	\item \code{oobError} the out-of-bag error.
      }
      \item \code{importance} if permution importance requested, an object of class \code{importanceCtg}, consisting of:
      \itemize{
	\item \code{mispred} the misprediction rate, by predictor.
	\item \code{oobErr} the out-of-bag error, by predictor.
      }
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
  rb <- arbTrain(pf, sp, y)


  # Performs separate prediction on new data:
  xx <- data.frame(replace(6, rnorm(nRow)))
  pred <- predict(rb, xx)
  yPred <- pred$yPred

  rb <- Rborist(x,y)

  # Performs separate prediction on new data:
  xx <- data.frame(replacate(6, rnorm(nRow)))
  pred <- predict(rb, xx)
  yPred <- pred$yPred

  # As above, but also records final indices of each tree walk:
  #
  pred <- predict(rb, xx, indexing=TRUE)
  print(pred$indices[c(1:2), ])


  # As above, but predicts over \code{newdata} with unobserved values.
  # In the case of numerical data, only missing values are considered
  # unobserved.  Missing values are encoded as \code{NaN}, which are
  # incomparable, precipitating \code{false} on every test.  Prediction
  # therefore takes the \code{false} branch when encountering missing
  # values:
  #
  xxMissing <- xx
  xxMissing[6, c(15, 32, 87, 101)] <- NA
  pred <- predict(rb, xxMissing)
  

  # As above, but returns a nonterminal score upon encountering
  # unobserved values. Neither the true nor the false branch from the
  # testing node is taken.  Instead, the score returned is derived
  # from all leaf nodes (terminals) reached by the testing
  # (nonterminal) node.
  #
  pred <- predict(rb, xxMissing, trapUnobserved = TRUE)


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

  # Using the \code{keyedFrame} option allows the columns of
  # \code{newdata} to appear in arbitrary order, so long as the
  # columns present during training appear as a subset:
  #
  pred <- predict(rb, iris[c(2, 4, 3, 1)], keyedFrame=TRUE)


  # As above, but validation census to report class probabilities:
  pred <- predict(rb, iris[-5], ctgCensus="prob")
  prob <- pred$prob


  # As above, but with training reponse as test vector:
  pred <- predict(rb, iris[-5], iris[5], ctgCensus = "prob")
  prob <- pred$prob
  conf <- pred$confusion
  misPred <- pred$misPred

  # As above, but predicts nonterminal when encountering categories
  # not observed during training.  That is, prediction returns a score
  # derived from all terminal nodes (leaves) reached from the
  # (nonterminal) testing node.
  #
  # In this case, "unobserved" refers to categories not present in
  # the subpartition over which a splitting is performed.  As training
  # partitions the data into smaller and smaller regions, a given
  # category becomes less likely to appear in a region.
  #
  # More generally, unobserved data can include missing predictors as
  # well as categories appearing in \code{newdata} which were not
  # present during training.
  #
  pred <- predict(rb, trapUnobserved=TRUE)
}
}

\author{
  Mark Seligman at Suiji.
}

\seealso{\code{\link{rfTrain}}}
  
    