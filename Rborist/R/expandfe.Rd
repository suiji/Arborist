% File man/expand.Rborist.Rd
% Part of the rborist package

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
  \item{arbOut}{an object of type \code{Rborist} produced by training.}
}

\value{An object of type \code{ExportReg} or \code{ExportCtg}.
}


\examples{
  \dontrun{
    data(iris)
    rb <- Rborist(iris[,-5], iris[,5])
    ffe <- export(rb)
  }
}

\author{
  Mark Seligman at Suiji.
}
