// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file predictscorer.h

   @brief Scorer for prediction.

   @author Mark Seligman
 */

#ifndef FOREST_PREDICTSCORER_H
#define FOREST_PREDICTSCORER_H


#include "typeparam.h"
#include "predict.h"

#include <vector>

class PredictScorer {
  const double nu; ///< Learning rate, possibly vector if adaptive.
  const double baseScore; ///< Pre-training score of full sample set.
  const CtgT nCtg; ///< Ultimately obtainable from baseScore.
  const double defaultPrediction; ///< Ultimately obtainable from baseScore.
  const class Predict* predict; ///< Null, if training.

 public:

  PredictScorer(const struct ScoreDesc* scoreDesc,
		const class Sampler* sampler,
		const class Predict* predict_);


  /**
     @brief Derives a mean prediction value for an observation.
   */
  double predictMean(size_t obsIdx) const;

  
  /**
     @brief Derives a summation.

     @return sum of predicted responses plus rootScore.
   */
  double predictSum(size_t obsIdx) const;


  CtgT predictProb(size_t obsIdx,
		   class CtgProb* ctgProb,
		   unsigned int census[]) const;
  
  
  PredictorT predictPlurality(size_t obsIdx,
			      unsigned int* census) const;
  
  
  PredictorT argMaxJitter(const unsigned int* census,
			  const vector<double>& ctgJitter) const;
};

#endif
