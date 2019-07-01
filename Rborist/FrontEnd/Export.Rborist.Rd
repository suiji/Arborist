% File man/export.Rborist.Rd
% Part of the rborist package

\name{Export}
\alias{Export}
\concept{decision trees}
\title{Exportation Format for Rborist Training Output}
\description{
  Formats training output into a form suitable for packages offering feature-contribution display.
}


\usage{
 \method{Export}{Rborist}(arbOut)
}

\arguments{
  \item{arbOut}{an object of type \code{Rborist} produced by training.}
}

\value{An object of type \code{RboristExport}, potentially suitable for
  packages offering feature-contribution display.
}


\examples{
  \dontrun{
    data(iris)
    rb <- Rborist(iris[,-5], iris[,5])
    rbExport <- Export(rb)
  }
}

\author{
  Mark Seligman at Suiji.
}
