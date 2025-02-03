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
		   nThread = 0,
                   verbose=FALSE,
                   ...)
}

\arguments{
  \item{x}{the design frame expressed as either a \code{data.frame}
    object with numeric and/or \code{factor} columns or as a numeric
    or factor-valued matrix.}
  \item{nThread}{number of cores to run in parallel, if available.}
  \item{verbose}{indicates whether to output progress of
    preformatting.}
  \item{...}{unused.}
}

\value{an object of class \code{Deframe} consisting of:
  \itemize{
    \item \code{rleFrame} run-length encoded representation of class \code{RLEFrame} consisting of:
    \itemize{
      \item \code{rankedFrame} run-length encoded representation of class \code{RankedFrame} consisting of:
      \itemize{
	\item \code{nRow} the number of observations encoded.
	\item \code{runVal} the run-length encoded values.
	\item \code{runRow} the corresponding row indices.
	\item \code{rleHeight} the number of encodings, per predictor.
	\item \code{topIdx} the accumulated end index, per predictor.
      }
      \item \code{numRanked} packed representation of sorted numerical values of class \code{NumRanked} consisting of:
      \itemize{
	\item \code{numVal} distinct numerical values.
	\item \code{numHeight} value offset per predictor.
      }
      \item \code{facRanked} packed representation of sorted factor
  values of class \code{FacRanked} consisting of:
      \itemize{
	\item \code{facVal} distinct factor values, zero-based.
	\item \code{facHeight} value offset per predictor.
      }
    }
    \item \code{nRow} the number of training observations.
    \item \code{signature} an object of type \code{Signature} consisting of:
    \itemize{
      \item \code{predForm} predictor class names.
      \item \code{level} per-predictor levels, regardless whether realized.
      \item \code{factor} per-predictor realized levels.
      \item \code{colNames} predictor names.
      \item \code{rowNames} observation names.
    }
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
