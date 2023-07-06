% File man/Rborist.Rd
% Part of the Rborist package

\name{Rborist}
\alias{Rborist}
\alias{Rborist.default}
\concept{decision trees}
\title{Rapid Decision Tree Construction and Evaluation}
\description{
  Legacy entry for accelerated implementation of the
  Random Forest (trademarked name) algorithm.  Calls the suggested
  entry, \code{rfArb}.
}

\usage{
\method{Rborist}{default} (x,
              y,
              ...)
}

\arguments{
  \item{x}{ the design matrix expressed as a \code{PreFormat} object, as a
  \code{data.frame} object with numeric and/or \code{factor} columns or
  as a numeric matrix.}
  \item{y}{ the response (outcome) vector, either numerical or
  categorical.  Row count must conform with \code{x}.}
  \item{...}{specific to \code{rfArb}.}
}

\value{an object of class \code{rfArb}, as documented in command of the
  same name.}


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
}
}

\author{
  Mark Seligman at Suiji.
}
