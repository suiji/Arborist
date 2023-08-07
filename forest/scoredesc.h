// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file scoredesc.h

   @brief Algorithm-specific container parametrizing scoring.

   @author Mark Seligman
 */

#ifndef FOREST_SCOREDESC_H
#define FOREST_SCOREDESC_H


#include "typeparam.h"


struct ScoreDesc {
  double nu; ///< Possibly vector, for adjustable learning rate.
  double baseScore; ///< Pre-training score of full sample set.
  // May require an enumeration of algorithm type, as well.

  /**
     @brief Independent trees.
   */
  ScoreDesc() :
    nu(0.0),
    baseScore(0.0) {
  }


  /**
     @brief Sequential trees.

     @param baseScore_ is zero prior to training the first tree.
   */
  ScoreDesc(double nu_,
	    double baseScore_ = 0.0) :
    nu(nu_),
    baseScore(baseScore_) {
  }


  /**
     @brief Facilitates naive construction from front end.
   */
  ScoreDesc(const pair<double, double>& valPair) :
  nu(valPair.first),
    baseScore(valPair.second) {
  }
  
  
  ~ScoreDesc() = default;

  /*
    @brief Builds algorithm-specific scorer for response type.
   */
  unique_ptr<class PredictScorer> makePredictScorer(const class Sampler* sampler,
					     const class Predict* predict) const;
};

#endif
