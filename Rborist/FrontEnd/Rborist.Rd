% File man/Rborist.Rd
% Part of the Rborist package

\name{Rborist}
\alias{Rborist}
\alias{Rborist.default}
\concept{decision trees}
\title{Rapid Decision Tree Construction and Evaluation}
\description{
  Accelerated implementation of the Random Forest (trademarked name)
  algorithm.  Tuned for multicore and GPU hardware.  Bindable with most
  numerical front-end languages in addtion to R.  Invocation is
  similar to that provided by "randomForest" package.
}

\usage{
\method{Rborist}{default} (x,
                y,
                autoCompress = 0.25,
                ctgCensus = "votes",
                classWeight = NULL,
                maxLeaf = 0,
                minInfo = 0.01,
                minNode = ifelse(is.factor(y), 2, 3),
                nLevel = 0,
                nSamp = 0,
                nThread = 0,
                nTree = 500,
                noValidate = FALSE,
                predFixed = 0,
                predProb = 0.0,
                predWeight = NULL, 
                quantVec = NULL,
                quantiles = !is.null(quantVec),
                regMono = NULL,
                rowWeight = NULL,
                splitQuant = NULL,
                thinLeaves = ifelse(is.factor(y), TRUE, FALSE),
                treeBlock = 1,
                verbose = FALSE,
                withRepl = TRUE,
                ...)
}

\arguments{
  \item{x}{ the design matrix expressed as a \code{PreFormat} object, as a
  \code{data.frame} object with numeric and/or \code{factor} columns or
  as a numeric matrix.}
  \item{y}{ the response (outcome) vector, either numerical or
  categorical.  Row count must conform with \code{x}.}
  \item{autoCompress}{plurality above which to compress predictor values.}
  \item{ctgCensus}{report categorical validation by vote or by probability.}
  \item{classWeight}{proportional weighting of classification
    categories.}
  \item{maxLeaf}{maximum number of leaves in a tree.  Zero denotes no limit.}
  \item{minInfo}{information ratio with parent below which node does not split.}
  \item{minNode}{minimum number of distinct row references to split a node.}
  \item{nLevel}{maximum number of tree levels to train.  Zero denotes no
    limit.}
  \item{nSamp}{number of rows to sample, per tree.}
  \item{nThread}{suggests an OpenMP-style thread count.  Zero denotes
    the default processor setting.}
  \item{nTree}{ the number of trees to train.}
  \item{noValidate}{whether to train without validation.}
  \item{predFixed}{number of trial predictors for a split (\code{mtry}).}
  \item{predProb}{probability of selecting individual predictor as trial splitter.}
  \item{predWeight}{relative weighting of individual predictors as trial
    splitters.}
  \item{quantVec}{quantile levels to validate.}
  \item{quantiles}{whether to report quantiles at validation.}
  \item{regMono}{signed probability constraint for monotonic
    regression.}
  \item{rowWeight}{row weighting for initial sampling of tree.}
  \item{splitQuant}{(sub)quantile at which to place cut point for
  numerical splits}.
  \item{thinLeaves}{bypasses creation of export and quantile state in
  order to reduce memory footprint.}
  \item{treeBlock}{maximum number of trees to train during a single
    level (e.g., coprocessor computing).}
  \item{verbose}{indicates whether to output progress of training.}
  \item{withRepl}{whether row sampling is by replacement.}
  \item{...}{not currently used.}
}

\value{ an object of class \code{Rborist}, a list containing the
  following items:

  \item{forest}{ a list containing
    
    \code{forestNode}{ a vector of packed structures expressing splitting
      predictors, splitting values, successor node deltas and leaf indices.}

    \code{height}{ a vector of accumulated tree heights within \code{forestNode}.}

    \code{facSplit}{ a vector of splitting factor values.}

    \code{facHeight}{ a vector of accumulated tree heights positions within the splitting factor values.}
  }
  a list containing either of:
  \code{LeafReg}{ a list consisting of regression leaf data:
      
    \code{node}{ a packed structure expressing leaf scores and node counts.}

    \code{nodeHeight}{ a vector of accumulated tree heights within
      \code{node}.}

    \code{bagHeight}{ a vector of accumulated bag counts, per tree.}

    \code{bagSample}{ a vector of packed data structures, one per
      unique row sample, containing the row index and number of times sampled.}

    \code{yTrain}{ the training response.}
  }
  or
  \code{LeafCtg}{ a list consisting of classification leaf data:
      
    \code{node}{ a packed structure expressing leaf scores and node counts.}

    \code{nodeHeight}{ a vector of accumulated tree heights within
      \code{node}.}

    \code{bagHeight}{ a vector of accumulated bag counts, per tree.}

    \code{bagSample}{ a vector of packed data structures, one per
        unique row sample, containing the row index and number of times sampled.}
      \code{weight}{ a vector of per-category probabilities, one set for
        each sampled row.}    
    
      \code{levels}{ a vector of strings containing the training response levels.}
    }

  \item{bag}{ a list consisting of bagged row information:

    \code{raw}{ a packed bit matrix indicating whether a given row, tree
      pair is bagged.}

    \code{nRow}{ the number of rows employed in training.}

    \code{nTree}{ the number of trained trees.}
    
    \code{rowBytes}{ the row stride, in bytes.}
  }

  \item{training}{ a list containing information gleaned during training:

    \code{call}{ a string containing the original invocation.}

    \code{info}{ the information contribution of each predictor.}

    \code{version}{ the version of the Rborist package.}

    \code{diag}{ strings containing unspecified diagnostic notes and observations.}
  }
  \item{validation}{ a list containing the results of validation, if requested:
    
    \code{ValidReg}{ a list of validation results for regression:
      
      \code{yPred}{ vector containing the predicted response.}

      \code{mae}{ the mean absolute error of prediction.}
    
      \code{mse}{ the mean-square error of prediction.}

      \code{rsq}{ the r-squared statistic.}

      \code{qPred}{ matrix containing the prediction quantiles, if requested.}
    }
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
  # Regression example:
  nRow <- 5000
  x <- data.frame(replicate(6, rnorm(nRow)))
  y <- with(x, X1^2 + sin(X2) + X3 * X4) # courtesy of S. Welling.

  # Classification example:
  data(iris)

  # Generic invocation:
  rb <- Rborist(x, y)


  # Causes 300 trees to be trained:
  rb <- Rborist(x, y, nTree = 300)


  # Causes rows to be sampled without replacement:
  rb <- Rborist(x, y, withRepl=FALSE)


  # Causes validation census to report class probabilities:
  rb <- Rborist(iris[-5], iris[5], ctgCensus="prob")


  # Applies table-weighting to classification categories:
  rb <- Rborist(iris[-5], iris[5], classWeight = "balance")


  # Weights first category twice as heavily as remaining two:
  rb <- Rborist(iris[-5], iris[5], classWeight = c(2.0, 1.0, 1.0))


  # Does not split nodes when doing so yields less than a 2\% gain in
  # information over the parent node:
  rb <- Rborist(x, y, minInfo=0.02)


  # Does not split nodes representing fewer than 10 unique samples:
  rb <- Rborist(x, y, minNode=10)


  # Trains a maximum of 20 levels:
  rb <- Rborist(x, y, nLevel = 20)


  # Trains, but does not perform subsequent validation:
  rb <- Rborist(x, y, noValidate=TRUE)


  # Chooses 500 rows (with replacement) to root each tree.
  rb <- Rborist(x, y, nSamp=500)


  # Chooses 2 predictors as splitting candidates at each node (or
  # fewer, when choices exhausted):
  rb <- Rborist(x, y, predFixed = 2)  


  # Causes each predictor to be selected as a splitting candidate with
  # distribution Bernoulli(0.3):
  rb <- Rborist(x, y, predProb = 0.3) 


  # Causes first three predictors to be selected as splitting candidates
  # twice as often as the other two:
  rb <- Rborist(x, y, predWeight=c(2.0, 2.0, 2.0, 1.0, 1.0))


  # Causes (default) quantiles to be computed at validation:
  rb <- Rborist(x, y, quantiles=TRUE)
  qPred <- rb$validation$qPred


  # Causes specfied quantiles (deciles) to be computed at validation:
  rb <- Rborist(x, y, quantVec = seq(0.1, 1.0, by = 0.10))
  qPred <- rb$validation$qPred


  # Constrains modelled response to be increasing with respect to X1
  # and decreasing with respect to X5.
  rb <- Rborist(x, y, regMono=c(1.0, 0, 0, 0, -1.0, 0))


  # Causes rows to be sampled with random weighting:
  rb <- Rborist(x, y, rowWeight=runif(nRow))


  # Suppresses creation of detailed leaf information needed for
  # quantile prediction and external tools.
  rb <- Rborist(x, y, thinLeaves = TRUE)


  # Sets splitting position for predictor 0 to far left and predictor
  # 1 to far right, others to default (median) position.

  spq <- rep(0.5, ncol(x))
  spq[0] <- 0.0
  spq[1] <- 1.0
  rb <- Rborist(x, y, splitQuant = spq)
  }
}


\author{
  Mark Seligman at Suiji.
}
