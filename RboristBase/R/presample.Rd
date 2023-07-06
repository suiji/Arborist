% File man/presample.Rd
% Part of the Rborist package

\name{prsample}
\alias{presample}
\alias{presample.default}
\concept{bagging}
\title{Forest-wide Observation Sampling}
\description{
  Observations sampled for each tree to be trained.  In the case of the
  Random Forest algorithm, this is the bag.
}
\usage{
\method{presample}(y,rowWeight = NULL, nSamp = 0, nTree = 500, withRepl
= TRUE, verbose = FALSE)
}

\arguments{
  \item{y}{A vector to be sampled, typically the response.}
  \item{rowWeight}{Per-observation sampling weights.  Default is
    uniform.}
  \item{nSamp}{Size of sample draw.  Default draws \code{y's} length.}
  \item{nTree}{Number of samples to draw.}
  \item{withRepl}{true iff sampling is with replacement.}
  \item{verbose}{true iff tracing execution.}
}

\value{
  \item{an object of class \code{Sampler} consisting of:
    \code{yTrain}{The sampled vector.}
    \code{nSamp}{The sample sizes drawn.}
    \code{nTree}{The number of independent samples.}
    \code{samples}{A packed data structure encoding the observation
      index and corresponding sample count.}
    \code{hash}{A hashed digest of the data items.}
  }
}


