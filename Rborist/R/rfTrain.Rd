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
                classWeight = numeric(0),
                maxLeaf = 0,
                minInfo = 0.01,
                minNode = if (is.factor(y)) 2 else 3,
                nLevel = 0,
                nThread = 0,
                predFixed = 0,
                predProb = 0.0,
                predWeight = numeric(0),
                regMono = numeric(0),
                splitQuant = numeric(0),
                thinLeaves = FALSE,
                treeBlock = 1,
                verbose = FALSE,
                ...)
}

\arguments{
  \item{y}{ the response (outcome) vector, either numerical or
    categorical.}
  \item{preFormat}{Compressed, presorted representation of the predictor
    values. Row count must conform with \code{y}.}
  \item{sampler}{Compressed representation of the sampled response.}
  \item{autoCompress}{plurality above which to compress predictor values.}
  \item{ctgCensus}{report categorical validation by vote or by probability.}
  \item{classWeight}{proportional weighting of classification
    categories.}
  \item{maxLeaf}{maximum number of leaves in a tree.  Zero denotes no limit.}
  \item{minInfo}{information ratio with parent below which node does not split.}
  \item{minNode}{minimum number of distinct row references to split a node.}
  \item{nLevel}{maximum number of tree levels to train, including
    terminals (leaves).  Zero denotes no limit.}
  \item{nThread}{suggests an \code{OpenMP}-style thread count.  Zero denotes
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
  \item{...}{Not currently used.}
}

\value{an object of class \code{arbTrain}, containing:
  \itemize{
    \item \code{version} the version of the \code{Rborist} package used to train.
    \item \code{samplerHash} hash value of the \code{Sampler} object used to train.  Recorded for consistency of subsequent commands.
    \item \code{predInfo} a vector of forest-wide Gini (classification) or weighted
      variance (regression), by predictor.
    \item \code{forest} an object of class \code{Forest} containing:
    \itemize{
      \item \code{nTree} the number of trees trained.
      \item \code{node} an object of class \code{Node} consisting of:
      \itemize{
	\item \code{treeNode} forest-wide vector of packed node representations.
	\item \code{extent} per-tree node counts.
	\item \code{scores} numeric vector of scores, for all terminals and nonterminals.
	\item \code{factor} an object of class \code{Factor} consisting of:
	\itemize{
	  \item \code{facSplit} forest-wide vector of packed factor bits.
	  \item \code{extent} per-tree extent of factor bits.
	  \item \code{observed} forest-wide vector of observed factor bits.
	}
      }
      \item \code{Leaf} an object of class \code{Leaf} containing:
      \itemize{
	\item \code{extent} forest-wide vector of leaf populations, i.e., counts of unique samples.
	\item \code{index} forest-wide vector of sample indices.
      }
    }
    \item \code{diag} diagnostics accumulated over the training task.
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


  # Causes validation census to report class probabilities:
  rt <- rfTrain(iris[-5], iris[5], ctgCensus="prob")


  # Applies table-weighting to classification categories:
  rt <- rfTrain(iris[-5], iris[5], classWeight = "balance")


  # Weights first category twice as heavily as remaining two:
  rt <- rfTrain(iris[-5], iris[5], classWeight = c(2.0, 1.0, 1.0))


  # Does not split nodes when doing so yields less than a 2\% gain in
  # information over the parent node:
  rt <- rfTrain(y, preFormat, sampler, minInfo=0.02)


  # Does not split nodes representing fewer than 10 unique samples:
  rt <- rfTrain(y, preFormat, sampler, minNode=10)


  # Trains a maximum of 20 levels:
  rt <- rfTrain(y, preFormat, sampler, nLevel = 20)


  # Trains, but does not perform subsequent validation:
  rt <- rfTrain(y, preFormat, sampler, noValidate=TRUE)


  # Chooses 500 rows (with replacement) to root each tree.
  rt <- rfTrain(y, preFormat, sampler, nSamp=500)


  # Chooses 2 predictors as splitting candidates at each node (or
  # fewer, when choices exhausted):
  rt <- rfTrain(y, preFormat, sampler, predFixed = 2)  


  # Causes each predictor to be selected as a splitting candidate with
  # distribution Bernoulli(0.3):
  rt <- rfTrain(y, preFormat, sampler, predProb = 0.3) 


  # Causes first three predictors to be selected as splitting candidates
  # twice as often as the other two:
  rt <- rfTrain(y, preFormat, sampler, predWeight=c(2.0, 2.0, 2.0, 1.0, 1.0))


  # Constrains modelled response to be increasing with respect to X1
  # and decreasing with respect to X5.
  rt <- rfTrain(x, y, preFormat, sampler, regMono=c(1.0, 0, 0, 0, -1.0, 0))


  # Suppresses creation of detailed leaf information needed for
  # quantile prediction and external tools.
  rt <- rfTrain(y, preFormat, sampler, thinLeaves = TRUE)

  spq <- rep(0.5, ncol(x))
  spq[0] <- 0.0
  spq[1] <- 1.0
  rt <- rfTrain(y, preFormat, sampler, splitQuant = spq)
  }
}


\author{
  Mark Seligman at Suiji.
}


\seealso{\code{\link{Rborist}}}
