% File man/rborist.Rd
% Part of the rborist package

\name{Rborist}
\alias{Rborist}
\alias{Rborist.default}
\concept{decision trees}
\title{Rapid Decision Tree Construction and Evaluation}
\description{
  Accelerated implementation of the Random Forest (trademarked name)
  algorithm.  Tuned for multicore and GPU hardware.  Bindable with most
  numerical front-end languages in addtion to R.  Invocation is
  similar to that provided by "randomForest" package.
}

\usage{
\method{Rborist}{default} (x, y, nTree=500, withRepl = TRUE,
                ctgCensus = "votes",
                classWeight = NULL,
                minInfo = 0.01,
                minNode = ifelse(is.factor(y), 2, 5),
                nLevel = 0,
                noValidate = FALSE,
                nSamp = 0,
                predFixed = 0,
                predProb = 0.0,
                predWeight = NULL, 
                qVec = NULL,
                quantiles = !is.null(qVec),
                qBin = 5000,
                regMono = NULL,
                rowWeight = NULL,
                treeBlock = 1,
                pvtBlock = 8, ...)
}

\arguments{
  \item{x}{ the design matrix expressed as a \code{PreTrain} object, as a
  \code{data.frame} object with numeric and/or \code{factor} columns or
  as a numeric matrix.}
  \item{y}{ the response (outcome) vector, either numerical or
  categorical.  Row count must conform with \code{x}.}
  \item{nTree}{ the number of trees to train.}
  \item{withRepl}{whether row sampling is by replacement.}
  \item{ctgCensus}{report categorical validation by vote or by probability.}
  \item{classWeight}{proportional weighting of classification categories.}
  \item{minInfo}{information ratio with parent below which node does not split.}
  \item{minNode}{minimum number of distinct row references to split a node.}
  \item{nLevel}{maximum number of tree levels to train.  Zero denotes no
    limit.}
  \item{noValidate}{whether to train without validation.}
  \item{nSamp}{number of rows to sample, per tree.}
  \item{predFixed}{number of trial predictors for a split (\code{mtry}).}
  \item{predProb}{probability of selecting individual predictor as trial splitter.}
  \item{predWeight}{relative weighting of individual predictors as trial
    splitters.}
  \item{qVec}{quantile levels to validate.}
  \item{quantiles}{whether to report quantiles at validation.}
  \item{qBin}{bin size for facilating quantiles at large sample count.}
  \item{regMono}{signed probability constraint for monotonic
    regression.}
  \item{rowWeight}{row weighting for initial sampling of tree.}
  \item{treeBlock}{maximum number of trees to train during a single
    level (e.g., coprocessor computing).}
  \item{pvtBlock}{maximum number of trees to train in a block (e.g.,
  cluster computing).}
  \item{...}{not currently used.}
}

\value{ an object of class \code{Rborist}, a list containing the
  following items:

  \item{forest}{ a list containing
    
    \code{forestNode}{ a vector of packed structures expressing splitting
    predictors, splitting values, successor node deltas and leaf indices.}

  \code{origin}{ a vector of tree starting positions within \code{forestNode}.}

  \code{facSplit}{ a vector of splitting factor values.}

  \code{facOrigin}{ a vector of tree starting positions within the
    splitting factor values.}
  }
  \item{leaf}{ a list containing either of
    
    \code{LeafReg}{ a list consisting of regression leaf data:
      
      \code{node}{ a packed structure expressing leaf scores and node counts.}

      \code{origin}{ a vector of tree starting positions within
	\code{node}.}

      \code{info}{ a list consisting of
	
	\code{rank}{ the rank of the response value corresponding to
	  indexed sample.}

	\code{sCount}{ the sample count corresponding to the indexed
	  sample.}
      }

      \code{yRanked}{ a sorted vector of response values.}
      }
    or
    \code{LeafCtg}{ a list consisting of classification leaf data:
      
      \code{node}{ a packed structure expressing leaf scores and node
	counts.}

      \code{origin}{ a vector of tree starting positions within
	\code{node}.}

      \code{info}{ a matrix of weighted scores, per category per sample.}
      
      \code{levels}{ a vector of strings containing the training response levels.}
      }
  }
  \item{signature}{ a list consisting of training information needed for
      separate testing and prediction:
    
    \code{nRow}{ the number of rows used to train.}
    
    \code{predMap}{ a vector mapping core predictor indices back to their
      respective front-end positions.}
    
    \code{level}{ a vector of strings vectors representing the training
    response factor levels.}
  }

  \item{training}{ a list containing information gleaned during training:
    
    \code{predInfo}{ the information contribution of each predictor.}
    
    \code{bag}{ a packed bit matrix representing the bagged samples in
      each tree.}
  }

  \item{validation}{ a list containing the results of validation:
    
    \code{ValidReg}{ a list of validation results for regression:
      
      \code{yPred}{ a vector containing the predicted response.}

      \code{mse}{ the mean-square error of prediction.}

      \code{rsq}{ the r-squared statistic.}

      \code{qPred}{ a matrix containing the prediction quantiles, if requested.}
    }

    \code{ValidCtg}{ a list of validation results for classification:
      
      \code{yPred}{ a vector containing the predicted response.}

      \code{misprediction}{ a vector containing the classwise
	misprediction rates.}
      
      \code{confusion}{ a confusion matrix.}

      \code{census}{ a matrix of predictions, by category.}

      \code{prob}{ a matrix of prediction probabilities by category, if requested.}
    }
  }
}



\author{
  Mark Seligman at Suiji.
}
