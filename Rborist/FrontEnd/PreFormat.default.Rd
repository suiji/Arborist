% File man/pretrain.Rd
% Part of the rborist package

\name{PreFormat}
\alias{PreFormat}
\alias{PreFormat.default}
\concept{decision trees}
\title{Preformatting for Training with Warm Starts}
\description{
  Presorts and formats training input into a form suitable for
  subsequent training by \code{Rborist} command.  Saves unnecessary
  recomputation of this form when iteratively retraining.
}


\usage{
\method{PreFormat}{default}(x, verbose)
}

\arguments{
  \item{x}{the design matrix expressed as either a \code{data.frame}
  object with numeric and/or \code{factor} columns or as a numeric
  matrix.}
  \item{verbose}{indicates whether to output progress of preformatting.}
}

\value{
  \item{PreFormat}{ a list consisting of three objects:

    \code{predFrame}{ a list of presorted and formatted predictor
    information:

    \code{blockFac}{ a matrix of zero-based factor predictor values.}

    \code{blockFacSparse}{ a sparse block of factor predictor values.}

    \code{blockNum}{ a matrix of dense numeric predictor values.}

    \code{blockNumSparse}{ a sparse block of numeric predictor values.}
    
    \code{nPredFac}{ the number of factor predictors.}

    \code{nRow}{ the number of training rows.}

    \code{nPredFac}{ the number of factor-valued predictors.}

    \code{facCard}{ a vector of the factor cardinalities.}

    \code{nPredNum}{ the number of numerical predictors.}
  }
  \code{summaryRLE}{ a run-length encoded version of the training
    observations.}
  \code{obsHash}{ a hashed version of the training observations, used to
  ensure that subsequent invocations of Rborist employ the same
  observations used to train.}
}
}


\examples{
  \dontrun{
    data(iris)
    pt <- PreFormat(iris[,-5])

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
