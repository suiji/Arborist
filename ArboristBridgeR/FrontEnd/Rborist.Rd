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
\method{Rborist}{default} (x, y, nTree=500, withRepl = TRUE,
                autoCompress = 0.25,
                ctgCensus = "votes",
                classWeight = NULL,
                maxLeaf = 0,
                minInfo = 0.01,
                minNode = ifelse(is.factor(y), 2, 3),
                nLevel = 0,
                noValidate = FALSE,
                nSamp = 0,
                predFixed = 0,
                predProb = 0.0,
                predWeight = NULL, 
                quantVec = NULL,
                quantiles = !is.null(quantVec),
                qBin = 5000,
                regMono = NULL,
                rowWeight = NULL,
                splitQuant = NULL,
                thinLeaves = FALSE,
                treeBlock = 1,
                pvtBlock = 8, ...)
}

\arguments{
  \item{x}{ the design matrix expressed as a \code{PreFormat} object, as a
  \code{data.frame} object with numeric and/or \code{factor} columns or
  as a numeric matrix.}
  \item{y}{ the response (outcome) vector, either numerical or
  categorical.  Row count must conform with \code{x}.}
  \item{nTree}{ the number of trees to train.}
  \item{withRepl}{whether row sampling is by replacement.}
  \item{autoCompress}{plurality above which to compress predictor values.}
  \item{ctgCensus}{report categorical validation by vote or by probability.}
  \item{classWeight}{proportional weighting of classification
    categories.}
  \item{maxLeaf}{maximum number of leaves in a tree.  Zero denotes no limit.}
  \item{minInfo}{information ratio with parent below which node does not split.}
  \item{minNode}{minimum number of distinct row references to split a node.}
  \item{nLevel}{maximum number of tree levels to train.  Zero denotes no
    limit.}
  \item{noValidate}{whether to train without validation.}
  \item{nSamp}{number of rows to sample, per tree.}
  \item{predFixed}{number of trial predictors for a split (\code{mtry}).}
  \item{predProb}{probability of selecting individual predictor as trial splitter.}
  \item{predWeight}{relative weighting of individual predictors as trial
    splitters.}
  \item{quantVec}{quantile levels to validate.}
  \item{quantiles}{whether to report quantiles at validation.}
  \item{qBin}{bin size for facilating quantiles at large sample count.}
  \item{regMono}{signed probability constraint for monotonic
    regression.}
  \item{rowWeight}{row weighting for initial sampling of tree.}
  \item{splitQuant}{(sub)quantile at which to place cut point for
  numerical splits}.
  \item{thinLeaves}{bypasses creation of export and quantile state in
  order to reduce memory footprint.}
  \item{treeBlock}{maximum number of trees to train during a single
    level (e.g., coprocessor computing).}
  \item{pvtBlock}{maximum number of trees to train in a block (e.g.,
    cluster computing).}
  \item{...}{not currently used.}
}

\value{ an object of class \code{Rborist}, a list containing the
  following items:

  \item{forest}{ a list containing
    
    \code{forestNode}{ a vector of packed structures expressing splitting
    predictors, splitting values, successor node deltas and leaf indices.}

  \code{origin}{ a vector of tree starting positions within \code{forestNode}.}

  \code{facSplit}{ a vector of splitting factor values.}

  \code{facOrigin}{ a vector of tree starting positions within the
    splitting factor values.}
  }
  \item{leaf}{ a list containing either of

    \code{BagRow}{ a vector of packed data structures, one per unique row
    sample, containing a row index and the number of times the row is sampled.}
    
    \code{LeafReg}{ a list consisting of regression leaf data:
      
      \code{node}{ a packed structure expressing leaf scores and node counts.}

      \code{origin}{ a vector of tree starting positions within
	\code{node}.

      \code{rank}{ a vector, one element per unique row sample, indicating
	  the rank of the sampled row.}
      }

      \code{yRanked}{ a sorted vector of response values.}
   }
    or
    \code{LeafCtg}{ a list consisting of classification leaf data:
      
      \code{node}{ a packed structure expressing leaf scores and node
	counts.}

      \code{origin}{ a vector of tree starting positions within
	\code{node}.}

      \code{weight}{ a matrix of weighted scores, per category per sample.}
      
      \code{levels}{ a vector of strings containing the training response levels.}
      }
  }
  \item{signature}{ a list consisting of training information needed for
      separate testing and prediction:
    
    \code{nRow}{ the number of rows used to train.}
    
    \code{predMap}{ a vector mapping core predictor indices back to their
      respective front-end positions.}
    
    \code{level}{ a vector of strings vectors representing the training
    response factor levels.}
  }

  \item{training}{ a list containing information gleaned during training:

    \code{call}{ a string containing the original invocation.}

    \code{info}{ the information contribution of each predictor.}

    \code{version}{ the version of the Rborist package.}

    \code{diag}{ strings containing unspecified diagnostic notes and observations.}
  }

  \item{validation}{ a list containing the results of validation:
    
    \code{ValidReg}{ a list of validation results for regression:
      
      \code{yPred}{ a vector containing the predicted response.}

      \code{mse}{ the mean-square error of prediction.}

      \code{mae}{ the mean absolute error of prediction.}
      
      \code{rsq}{ the r-squared statistic.}

      \code{qPred}{ a matrix containing the prediction quantiles, if requested.}
    }

    \code{ValidCtg}{ a list of validation results for classification:
      
      \code{yPred}{ a vector containing the predicted response.}

      \code{misprediction}{ a vector containing the classwise
	misprediction rates.}

      \code{oobError}{ the out-of-bag error rate}
      
      \code{confusion}{ a confusion matrix.}

      \code{census}{ a matrix of predictions, by category.}

      \code{prob}{ a matrix of prediction probabilities by category, if requested.}
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


  # Causes (default) quantile computation to be approximated by a
  # small bin size of 100:  fast, but not as accurate:
  rb <- Rborist(x, y, quantiles = TRUE, qBin = 100).
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
