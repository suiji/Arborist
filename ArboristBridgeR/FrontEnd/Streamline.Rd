% File man/Streamline.Rd
% Part of the Rborist package

\name{Streamline}
\alias{Streamline}
\alias{Streamline.default}
\concept{decision forest simplification}
\title{Reducing Memory Footprint of Trained Decision Forest}
\description{
  Clears fields deemed no longer useful.
}

\usage{
\method{Streamline}{default}(rb)
}

\arguments{
  \item{rb}{Trained forest object.}
}

\value{an object of class \code{Rborist} with certain fields cleared.
}



\examples{
\dontrun{
    ## Trains without validation.
    rb <- Rborist(x, y, noValidate=TRUE)
    ...
    ## Delayed validation using a PreFormat object.
    rb <- Streamline(rb)
 }
}


\author{
  Mark Seligman at Suiji.
}
