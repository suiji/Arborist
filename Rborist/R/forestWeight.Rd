% File man/forestWeight.Rd
% Part of the Rborist package

\name{forestWeight}
\alias{forestWeight}
\alias{forestWeight.default}
\title{Meinshausen forest weights}
\description{
  Normalized observation counts across a prediction set.
}

\usage{
\method{forestWeight}{default}(objTrain, prediction, sampler=objTrain$sampler,
nThread=0, verbose = FALSE, ...)
}

\arguments{
  \item{objTrain}{an object of class \code{rfArb}, created from a
    previous invocation of the command \code{Rborist} or \code{rfArb} to train.}
  \item{prediction}{an object of class \code{SummaryReg} or
    \code{SummaryCtg} obtained from prediction using \code{objTrain} and
    argument \code{indexing=TRUE}.}
  \item{sampler}{an object of class \code{Sampler}, as documented in
  command of the same name.}
  \item{nThread}{specifies a prefered thread count.}
  \item{verbose}{whether to output progress of weighting.}
  \item{...}{not currently used.}
}

\value{a numeric matrix having rows equal to the Meinshausen weight of each new datum.}


\examples{
\dontrun{
  # Regression example:
  nRow <- 5000
  x <- data.frame(replicate(6, rnorm(nRow)))
  y <- with(x, X1^2 + sin(X2) + X3 * X4) # courtesy of S. Welling.
  rb <- Rborist(x,y)


  newdata <- data.frame(replace(6, rnorm(nRow)))

  # Performs separate prediction on new data, saving indices:
  pred <- predict(rb, newdata, indexing=TRUE)
  weights <- forestWeight(rb, pred)

  obsIdx <- 215 # Arbitrary observation index (row number)

  # Inner product should equal prediction, modulo numerical vagaries:
  yPredApprox <- weights[obsIdx,] \%*\% y
  print((yPredApprox - pred$yPred[obsIdx])/yPredApprox) 
}

}

\author{
  Mark Seligman at Suiji.
}

\seealso{\code{\link{Rborist}}}
  
    