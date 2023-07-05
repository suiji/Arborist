% File man/rfTrain.Rd
% Part of the Rborist package

\name{rfTrain}
\alias{rfTrain}
\alias{rfTrain.default}
\concept{decision trees}
\title{Rapid Decision Tree Training}
\description{
  Accelerated training using the Random Forest (trademarked name)
  algorithm.  Tuned for multicore and GPU hardware.  Bindable with most
  numerical front-end languages in addtion to R.
}

\usage{
\method{rfTrain}{default}(preFormat,
                 sampler,
                 y,
                autoCompress = 0.25,
                ctgCensus = "votes",
                classWeight = NULL,
                maxLeaf = 0,
                minInfo = 0.01,
                minNode = if (is.factor(y)) 2 else 3,
                nLevel = 0,
                nThread = 0,
                predFixed = 0,
                predProb = 0.0,
                predWeight = NULL, 
                regMono = NULL,
                rowWeight = NULL,
                splitQuant = NULL,
                thinLeaves = FALSE,
                treeBlock = 1,
                verbose = FALSE)
}

\arguments{
  \item{y}{ the response (outcome) vector, either numerical or
    categorical.}
  \item{preformat}{Compressed, presorted representation of the predictor
    values. Row count must conform with \code{y}.}
  \item{sampler}{Compressed representation of the sampled response.}
  \item{autoCompress}{plurality above which to compress predictor values.}
  \item{ctgCensus}{report categorical validation by vote or by probability.}
  \item{classWeight}{proportional weighting of classification
    categories.}
  \item{maxLeaf}{maximum number of leaves in a tree.  Zero denotes no limit.}
  \item{minInfo}{information ratio with parent below which node does not split.}
  \item{minNode}{minimum number of distinct row references to split a node.}
  \item{nLevel}{maximum number of tree levels to train.  Zero denotes no
    limit.}
  \item{nThread}{suggests an OpenMP-style thread count.  Zero denotes
    the default processor setting.}
  \item{predFixed}{number of trial predictors for a split (\code{mtry}).}
  \item{predProb}{probability of selecting individual predictor as trial splitter.}
  \item{predWeight}{relative weighting of individual predictors as trial
    splitters.}
  \item{regMono}{signed probability constraint for monotonic
    regression.}
  \item{splitQuant}{(sub)quantile at which to place cut point for
    numerical splits}.
  \item{thinLeaves}{bypasses creation of leaf state in order to reduce
    memory footprint.}
  \item{treeBlock}{maximum number of trees to train during a single
    level (e.g., coprocessor computing).}
  \item{verbose}{indicates whether to output progress of training.}
  \item{withRepl}{whether row sampling is by replacement.}
}

\value{ an object of class \code{rfTrain}, a list containing the
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
      
    \code{score}{ a vector of leaf scores.}

    \code{sampler}{ a vector of packed data structures, one per
      unique row sample, containing the row index and number of times sampled.}

    \code{yTrain}{ the training response.}
  }
  or
  \code{LeafCtg}{ a list consisting of classification leaf data:
      
    \code{score}{ a vector of leaf scores.}

    \code{sampler}{ a vector of packed data structures, one per
        unique row sample, containing the leaf index and number of times sampled.}
      \code{weight}{ a vector of per-category probabilities, one set for
        each sampled row.}    
    
      \code{levels}{ a vector of strings containing the training response levels.}
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
  rt <- rfTrain(y)


  # Causes 300 trees to be trained:
  rt <- rfTrain(y, nTree = 300)


  # Causes rows to be sampled without replacement:
  rt <- rfTrain(x, y, withRepl=FALSE)


  # Causes validation census to report class probabilities:
  rt <- rfTrain(iris[-5], iris[5], ctgCensus="prob")


  # Applies table-weighting to classification categories:
  rt <- rfTrain(iris[-5], iris[5], classWeight = "balance")


  # Weights first category twice as heavily as remaining two:
  rt <- rfTrain(iris[-5], iris[5], classWeight = c(2.0, 1.0, 1.0))


  # Does not split nodes when doing so yields less than a 2\% gain in
  # information over the parent node:
  rt <- rfTrain(y, preformat, sampler, minInfo=0.02)


  # Does not split nodes representing fewer than 10 unique samples:
  rt <- rfTrain(y, preformat, sampler, minNode=10)


  # Trains a maximum of 20 levels:
  rt <- rfTrain(y, preformat, sampler, nLevel = 20)


  # Trains, but does not perform subsequent validation:
  rt <- rfTrain(y, preformat, sampler, noValidate=TRUE)


  # Chooses 500 rows (with replacement) to root each tree.
  rt <- rfTrain(y, preformat, sampler, nSamp=500)


  # Chooses 2 predictors as splitting candidates at each node (or
  # fewer, when choices exhausted):
  rt <- rfTrain(y, preformat, sampler, predFixed = 2)  


  # Causes each predictor to be selected as a splitting candidate with
  # distribution Bernoulli(0.3):
  rt <- rfTrain(y, preformat, sampler, predProb = 0.3) 


  # Causes first three predictors to be selected as splitting candidates
  # twice as often as the other two:
  rt <- rfTrain(y, preformat, sampler, predWeight=c(2.0, 2.0, 2.0, 1.0, 1.0))


  # Constrains modelled response to be increasing with respect to X1
  # and decreasing with respect to X5.
  rt <- rfTrain(x, y, preformat, sampler, regMono=c(1.0, 0, 0, 0, -1.0, 0))


  # Causes rows to be sampled with random weighting:
  rt <- rfTrain(y, preformat, sampler, rowWeight=runif(nRow))


  # Suppresses creation of detailed leaf information needed for
  # quantile prediction and external tools.
  rt <- rfTrain(y, preformat, sampler, thinLeaves = TRUE)

  spq <- rep(0.5, ncol(x))
  spq[0] <- 0.0
  spq[1] <- 1.0
  rt <- rfTrain(y, preformat, sampler, splitQuant = spq)
  }
}


\author{
  Mark Seligman at Suiji.
}
