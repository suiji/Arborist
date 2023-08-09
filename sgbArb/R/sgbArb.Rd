% File man/sgbArb.Rd
% Part of the Rborist package

\name{sgbArb}
\alias{sgbArb}
\alias{sgbArb.default}
\concept{decision trees}
\title{Stochastic Gradient Boosting with Rapid Training and Prediction}
\description{
  Accelerated implementation of Friedman's stochastic gradient boosting
  algorithm.  Tuned for multicore and GPU hardware.  Bindable with most
  numerical front-end languages in addtion to R.  Invocation is
  similar to that provided by \code{Rborist} package.
}

\usage{
\method{sgbArb}{default} (x, y,
                autoCompress = 0.25,            
                 impPermute = 0,
                indexing = FALSE,
                maxLeaf = 0,
                minInfo = 0.01,
                minNode = if (is.factor(y)) 2 else 3,
                nLevel = 6,
                nSamp = length(y),
                nThread = 0,
                nTree = 100,
                withRepl = FALSE,
                noValidate = (!withRepl || (nSamp == length(y))),
                nu = 0.1,
                predFixed = 0,
                predProb = 0.0,
                predWeight = NULL, 
                regMono = NULL,
                rowWeight = NULL,
                splitQuant = NULL,
                thinLeaves = is.factor(y) && !indexing,
                trapUnobserved = FALSE,
                treeBlock = 1,
                verbose = FALSE,
                ...)
}

\arguments{
  \item{x}{ the design matrix expressed as a \code{PreFormat} object, as a
  \code{data.frame} object with numeric and/or \code{factor} columns or
  as a numeric matrix.}
  \item{y}{ the response (outcome) vector, either numerical or
    categorical.  Row count must conform with \code{x}.}
  \item{autoCompress}{plurality above which to compress predictor values.}
  \item{impPermute}{number of importance permutations:  0 or 1.}
  \item{indexing}{whether to report final index, typically terminal, of
    tree traversal.}
  \item{maxLeaf}{maximum number of leaves in a tree.  Zero denotes no limit.}
  \item{minInfo}{information ratio with parent below which node does not split.}
  \item{minNode}{minimum number of distinct row references to split a node.}
  \item{nLevel}{maximum number of tree levels to train, including the root.  Zero denotes no
    limit.}
  \item{nSamp}{number of rows to sample, per tree.}
  \item{nThread}{suggests an OpenMP-style thread count.  Zero denotes
    the default processor setting.}
  \item{nTree}{the number of trees to train.}
  \item{nu}{is the learning rate.}
  \item{noValidate}{whether to train without validation.}
  \item{predFixed}{number of trial predictors for a split (\code{mtry}).}
  \item{predProb}{probability of selecting individual predictor as trial splitter.}
  \item{predWeight}{relative weighting of individual predictors as trial
    splitters.}
  \item{regMono}{signed probability constraint for monotonic
    regression.}
  \item{rowWeight}{row weighting for initial sampling of tree.}
  \item{splitQuant}{(sub)quantile at which to place cut point for
    numerical splits}.
  \item{thinLeaves}{bypasses creation of leaf state in order to reduce
    memory footprint.}
  \item{trapUnobserved}{reports score for nonterminal upon encountering
  values not observed during training, such as missing data.}
  \item{treeBlock}{maximum number of trees to train during a single
    level (e.g., coprocessor computing).}
  \item{verbose}{indicates whether to output progress of training.}
  \item{withRepl}{whether row sampling is by replacement.}
  \item{...}{not currently used.}
}

\value{ an object of class \code{sgbArb}, a list containing the
  following items:
  \item{sampler}{An object of class \code{Sampler}, as described in the
  documentation for the \code{presample} command, that summarizes the
  bag.}
  \item{leaf}{An object of class \code{Leaf}, as described in the
  documentation for the command \code{rfTrain}, that summarizes the
  terminal nodes of each tree.}
  \item{forest}{An object of class \code{Forest}, as described in the
  documentation for the command \code{rfTrain}, that summarizes the
  trained decision trees.}
  \item{predMap}{A vector of integers mapping internal to front-end
    predictor indices.}
  \item{signature}{An object of class \code{Signature}, as described
  inthe documentation for the command{presample}, that summarizes the
  training frame}
  \item{training}{A list summarizing the training task, consisting of
    the following fields:
    \code{call}{The calling invocation.}
    \code{info}{A vector of forest-wide Gini (classification) or weighted
      variance (regression), by predictor.}
    \code{version}{The version of the \code{Rborist} package used to train.}
    \code{diag}{Diagnostics accumulated over the training task.}
    \code{samplerHash}{Hash value of the \code{Sampler} object used to
      train.  Recorded for consistency of subsequent commands.}
  }
  \item{prediction}{An object of class \code{PredictReg} or
  \code{PredictCtg}, as described by the documention for command \code{predict}.}
  \item{validation}{An object of class \code{ValidReg} or
  \code{ValidCtg}, as described by the documention for command
  \code{predict}, if validation is requested.}
  \item{importance}{An object of class \code{ImportanceReg} or
  \code{ImportanceCtg}, as described by the documention for command
  \code{predict}, if permutation performance has been requested.}
}


\examples{
\dontrun{
  # Regression example:
  nRow <- 5000
  x <- data.frame(replicate(6, rnorm(nRow)))
  y <- with(x, X1^2 + sin(X2) + X3 * X4) # courtesy of S. Welling.


  # Generic invocation:
  rb <- sgbArb(x, y)


  # Causes 300 trees to be trained:
  rb <- sgbArb(x, y, nTree = 300)


  # Causes rows to be sampled without replacement:
  rb <- sgbArb(x, y, withRepl=FALSE)


  # Does not split nodes when doing so yields less than a 2\% gain in
  # information over the parent node:
  rb <- sgbArb(x, y, minInfo=0.02)


  # Does not split nodes representing fewer than 10 unique samples:
  rb <- sgbArb(x, y, minNode=10)


  # Trains a maximum of 20 levels:
  rb <- sgbArb(x, y, nLevel = 20)


  # Trains, but does not perform subsequent validation:
  rb <- sgbArb(x, y, noValidate=TRUE)


  # Chooses 500 rows (with replacement) to root each tree.
  rb <- sgbArb(x, y, nSamp=500)


  # Chooses 2 predictors as splitting candidates at each node (or
  # fewer, when choices exhausted):
  rb <- sgbArb(x, y, predFixed = 2)  


  # Causes each predictor to be selected as a splitting candidate with
  # distribution Bernoulli(0.3):
  rb <- sgbArb(x, y, predProb = 0.3) 


  # Causes first three predictors to be selected as splitting candidates
  # twice as often as the other two:
  rb <- sgbArb(x, y, predWeight=c(2.0, 2.0, 2.0, 1.0, 1.0))


  # Constrains modelled response to be increasing with respect to X1
  # and decreasing with respect to X5.
  rb <- sgbArb(x, y, regMono=c(1.0, 0, 0, 0, -1.0, 0))


  # Causes rows to be sampled with random weighting:
  rb <- sgbArb(x, y, rowWeight=runif(nRow))


  # Suppresses creation of detailed leaf information needed for
  # quantile prediction and external tools.
  rb <- sgbArb(x, y, thinLeaves = TRUE)

  # Directs prediction to take a random branch on encountering
  # values not observed during training, such as NA or an
  # unrecognized category.

  predict(rb, trapUnobserved = FALSE)

  # Directs prediction to silently trap unobserved values, reporting a
  # score associated with the current nonterminal tree node.

  predict(rb, trapUnobserved = TRUE)

  # Sets splitting position for predictor 0 to far left and predictor
  # 1 to far right, others to default (median) position.

  spq <- rep(0.5, ncol(x))
  spq[0] <- 0.0
  spq[1] <- 1.0
  rb <- sgbArb(x, y, splitQuant = spq)
  }
}

\author{
  Mark Seligman at Suiji.
}

\seealso{\code{\link{sgbArb}}}
