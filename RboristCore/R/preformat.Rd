% File man/preformat.Rd
% Part of the rborist package

\name{preformat}
\alias{preformat}
\alias{preformat.default}
\concept{decision trees}
\title{Preformatting for Training with Warm Starts}
\description{
  Presorts and formats training input into a form suitable for
  subsequent training by \code{Rborist} command.  Saves unnecessary
  recomputation of this form when iteratively retraining, for example
  when training parameter sweeps.
}


\usage{
\method{preformat}{default}(x, verbose=FALSE, ...)
}

\arguments{
  \item{x}{the design matrix expressed as either a \code{data.frame}
  object with numeric and/or \code{factor} columns or as a numeric
  matrix.}
\item{verbose}{indicates whether to output progress of preformatting.}
\item{...}{unused.}
}

\value{
  \item{preformat}{ a list consisting of three objects:

    \code{rleFrame}{ a run-length encoded representation of the observations.}

    \code{nRow}{ the number of training rows.}

    \code{signature}{ a list of predictor characteristics.}
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
