% File man/Export.Rborist.Rd
% Part of the rborist package

\name{Export}
\alias{Export}
\alias{Export.Rborist}
\concept{decision trees}
\title{Exportation Format for Rborist Training Output}
\description{
  Formats training output into a form suitable for illustration of
  feature contributions.
}


\usage{
 \method{Export}{Rborist}(arbOut)
}

\arguments{
  \item{arbOut}{an object of type \code{Rborist} produced by training.}
}

\value{An object of type \code{Export}.
}


\examples{
  \dontrun{
    data(iris)
    rb <- Rborist(iris[,-5], iris[,5])
    ffe <- Export(rb)
  }
}

\author{
  Mark Seligman at Suiji.
}
