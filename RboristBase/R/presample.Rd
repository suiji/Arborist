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
                            samplingWeight = numeric(0),
                            nSamp = 0,
                            nRep = 500,
                            withRepl =  TRUE,
                            nHoldout = 0,
                            nFold = 1,
                            verbose = FALSE,
                            nTree = 0,
                            ...)
}

\arguments{
  \item{y}{A vector to be sampled, typically the response.}
  \item{samplingWeight}{Per-observation sampling weights.  Default is
    uniform.}
  \item{nSamp}{Size of sample draw.  Default draws \code{y} length.}
  \item{nRep}{Number of samples to draw.  Replaces deprecated \code{nTree}.}
  \item{withRepl}{true iff sampling is with replacement.}
  \item{nHoldout}{Number of observations to omit from sampling.
    Augmented by unobserved response values.}
  \item{nFold}{Number of collections into which to partition the
    respone.}
  \item{verbose}{true iff tracing execution.}
  \item{nTree}{Number of samples to draw.  Deprecated.}
  \item{...}{not currently used.}
}

\value{an object of class \code{Sampler} consisting of:
  \itemize{
    \item \code{yTrain} the sampled vector.
    \item \code{nSamp} the sample sizes drawn.
    \item \code{nRep} the number of independent samples.
    \item \code{nTree} synonymous with \code{nRep}.  Deprecated.
    \item \code{samples} a packed data structure encoding the observation
      index and corresponding sample count.
    \item \code{hash} a hashed digest of the data items.
  }
}


\examples{
  \dontrun{
    y <- runif(1000)

    # Samples with replacement, 500 vectors of length 1000:
    ps <- presample(y)

    # Samples, as above, with 63 observations held out:
    ps <- presample(y, nHoldout = 63)

    # Samples without replacement, 250 vectors of length 500:
    ps2 <- presample(y, nTree=250, nSamp=500, withRepl = FALSE)


  }
}

\references{
  Tille, Yves. Sampling algorithms. Springer New York, 2006.
}
