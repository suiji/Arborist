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
 \method{expandfe}{default}(arbOut)
}

\arguments{
  \item{arbOut}{an object of type \code{RfArb} produced by training.}
}

\value{An object of type \code{ExportReg} or \code{ExportCtg} containing
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
