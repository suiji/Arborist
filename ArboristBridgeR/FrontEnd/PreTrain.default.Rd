% File man/pretrain.Rd
% Part of the rborist package

\name{PreTrain}
\alias{PreTrain}
\alias{PreTrain.default}
\concept{decision trees}
\title{Preformatting for Training with Warm Starts}
\description{
  Presorts and formats training input into a form suitable for
  subsequent training by \code{Rborist} command.
}


\usage{
\method{PreTrain}{default}(x)
}

\arguments{
  \item{x}{the design matrix expressed as either a \code{data.frame}
  object with numeric and/or \code{factor} columns or as a numeric matrix.}
}

\value{
  An object of type \code{PreTrain}.
}


\author{
  Mark Seligman at Suiji.
}
