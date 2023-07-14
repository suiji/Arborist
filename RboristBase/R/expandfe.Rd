% File man/expandfe.Rborist.Rd
% Part of the Rborist package

\name{expandfe}
\alias{expandfe}
\alias{expandfe.default}
\concept{decision trees}
\title{Expands forest values into front-end readable vectors.}
\description{
  Formats training output into a form suitable for illustration of
  feature contributions.
}


\usage{
 \method{expandfe}{default}(arbOut, sampler=NULL)
}

\arguments{
  \item{arbOut}{an object of type \code{rfTrain} produced by training.}
  \item{sampler}{an object ot type \code{Sampler} used to train.}
}

\value{An object of type \code{ExpandReg} or \code{ExpandCtg} containing
  human-readable representations of the trained forest.
}


\examples{
  \dontrun{
    data(iris)
    rb <- Rborist(iris[,-5], iris[,5])
    ffe <- expandfe(rb)

    # An rfTrain counterpart is NYI.
  }
}

\author{
  Mark Seligman at Suiji.
}
