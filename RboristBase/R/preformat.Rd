% File man/preformat.Rd
% Part of the Rborist package

\name{preformat}
\alias{preformat}
\alias{preformat.default}
\concept{decision trees}
\title{Preformatting for Training with Warm Starts}
\description{
  Presorts and formats training frame into a form suitable for
  subsequent training by \code{rfArb} caller or \code{rfTrain}
  command.  Wraps this form to spare unnecessary recomputation when
  iteratively retraining, for example, under parameter sweep.
}


\usage{
\method{preformat}{default}(x,
                   verbose=FALSE,
                   ...)
}

\arguments{
  \item{x}{the design frame expressed as either a \code{data.frame}
    object with numeric and/or \code{factor} columns or as a numeric
    or factor-valued matrix.}
  \item{verbose}{indicates whether to output progress of
    preformatting.}
  \item{...}{unused.}
}

\value{an object of class \code{Deframe} consisting of:
  \item{rleFrame}{run-length encoded representation of class
    \code{RLEFrame} consisting of:
    \code{rankedFrame}{run-length encoded representation of class
      \code{RankedFrame} consisting of:
      \code{nRow}{The number of observations encoded.}
      \code{runVal}{The run-length encoded values.}
      \code{runRow}{The corresponding row indices.}
      \code{rleHeight}{The number of encodings, per predictor.}
      \code{topIdx}{The accumulated end index, per predictor.}
    }
    \code{numRanked}{packed representation of sorted numerical values
      of class \code{NumRanked} consisting of:
      \code{numVal}{distinct numerical values.}
      \code{numHeight}{value offset per predictor.}
    }
    \code{facRanked}{packed representation of sorted factor values
      of class \code{FacRanked} consisting of:
      \code{facVal}{distinct factor values, zero-based.}
      \code{facHeight}{value offset per predictor.}
    }
  }
  \item{nRow}{the number of training observations.}
  \item{signature}{an object of type \code{Signature} consisting of:
    \code{predForm}{Predictor class names.}
    \code{level}{Per-predictor levels, regardless whether realized.}
    \code{factor}{Per-predictor realized levels.}
    \code{colNames}{Predictor names.}
    \code{rowNames}{Observation names.}
  }
}


\examples{
  \dontrun{
    data(iris)
    pt <- preformat(iris[,-5])

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

\seealso{\code{\link{Rborist}}}
