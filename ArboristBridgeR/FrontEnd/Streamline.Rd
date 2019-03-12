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
\method{Streamline}{default}(rs)
}

\arguments{
  \item{rs}{Trained forest object.}
}

\value{an object of class \code{Rborist} with certain fields cleared.
}



\examples{
\dontrun{
    ## Trains.
    rs <- Rborist(x, y)
    ...
    ## Replaces trained object with streamlined copy.
    rs <- Streamline(rs)
 }
}


\author{
  Mark Seligman at Suiji.
}
