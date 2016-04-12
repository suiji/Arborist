% File man/pretrain.Rd
% Part of the rborist package

\name{PreTrain}
\alias{PreTrain}
\alias{PreTrain.default}
\concept{decision trees}
\title{Preformatting for Training with Warm Starts}
\description{
  Presorts and formats training input into a form suitable for
  subsequent training by \code{Rborist} command.  Saves unnecessary
  recomputation of this form when iteratively retraining.
}


\usage{
\method{PreTrain}{default}(x)
}

\arguments{
  \item{x}{the design matrix expressed as either a \code{data.frame}
  object with numeric and/or \code{factor} columns or as a numeric matrix.}
}

\value{
  \item{PreTrain}{ a list of presorted and formatted predictor
    information:

    \code{colNames}{ a vector of strings containing the training column
      names.}

    \code{rowNames}{ a vector of strings containing the training row
      names.}

    \code{blockNum}{ a matrix containing all numeric predictor values.}

    \code{blockFac}{ a matrix containing all factor predictor values.}

    \code{nPredFac}{ the number of factor predictors.}

    \code{nRow}{ the number of training rows.}

    \code{facCard}{ a vector of the factor cardinalities.}

    \code{signature}{ a list consisting of training information needed
      for separate testing and prediction:

      \code{nRow}{ the number of training rows.}

      \code{predMap}{ a vector mapping core predictor indices back to
	their respective front-end positions.}

      \code{level}{ a vector of strings containing the training response
      factor levels.}

    }
  }
}

\examples{
  \dontrun{
    data(iris)
    pt <- PreTrain(iris[,-5])

    ppTry <- seq(0.2, 0.5, by= 0.3/10)
    nIter <- length(ppTry)
    rsq <- numeric(nIter)
    for (i in 1:nIter) {
      rb <- Rborist(pt, iris[,5], predProb=ppTry[i])
      rsq[i] = rb$validiation$rsq
    }
  }
}


\author{
  Mark Seligman at Suiji.
}
