% File man/presample.Rd
% Part of the Rborist package

\name{presample}
\alias{presample}
\alias{presample.default}
\concept{bagging}
\title{Forest-wide Observation Sampling}
\description{
  Observations sampled for each tree to be trained.  In the case of the
  Random Forest algorithm, this is the bag.
}
\usage{
\method{presample}{default}(y,
                            rowWeight = NULL,
                            nSamp = 0,
                            nTree = 500,
                            withRepl =  TRUE,
                            verbose = FALSE,
                            ...)
}

\arguments{
  \item{y}{A vector to be sampled, typically the response.}
  \item{rowWeight}{Per-observation sampling weights.  Default is
    uniform.}
  \item{nSamp}{Size of sample draw.  Default draws \code{y} length.}
  \item{nTree}{Number of samples to draw.}
  \item{withRepl}{true iff sampling is with replacement.}
  \item{verbose}{true iff tracing execution.}
  \item{...}{not currently used.}
}

\value{an object of class \code{Sampler} consisting of:
  \item{yTrain}{The sampled vector.}
  \item{nSamp}{The sample sizes drawn.}
  \item{nTree}{The number of independent samples.}
  \item{samples}{A packed data structure encoding the observation
    index and corresponding sample count.}
  \item{hash}{A hashed digest of the data items.}
}

\examples{
  \dontrun{
    y <- runif(1000)

    # Samples with replacement, 500 vectors of length 1000:
    ps <- presample(y)

    # Samples without replacement, 250 vectors of length 500:
    ps2 <- presample(y, nTree=250, nSamp=500, withRepl = FALSE)
  }
}