% File man/forestFloorExport.Rborist.Rd
% Part of the rborist package

\name{ForestFloorExport}
\alias{ForestFloorExport}
\alias{ForestFloorExport.Rborist}
\concept{decision trees}
\title{Exportation Format for Rborist Training Output}
\description{
  Formats training output into a form suitable for \code{ForestFloor} feature-contribution package.
}


\usage{
 \method{ForestFloorExport}{Rborist}(arbOut)
}

\arguments{
  \item{arbOut}{an object of type \code{Rborist} produced by training.}
}

\value{An object of type \code{ForestFloorExport}, as specified by the
  interface for the \code{ForestFloor} package.
}


\examples{
  \dontrun{
    data(iris)
    rb <- Rborist(iris[,-5], iris[,5])
    ffe <- ForestFloorExport(rb)

    library(ForestFloor)
    ForestFloor(ffe)
  }
}

\author{
  Mark Seligman at Suiji.
}
