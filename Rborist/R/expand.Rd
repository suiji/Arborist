% File man/expand.Rborist.Rd
% Part of the rborist package

\name{expand}
\alias{expand}
\alias{expand.default}
\concept{decision trees}
\title{Expands forest values into separate vectors.}
\description{
  Formats training output into a form suitable for illustration of
  feature contributions.
}


\usage{
 \method{expand}{default}(arbOut)
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
