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

#include <vector>
#include <string>


/**
   @brief Advises prediction how to derive a forest-wide score.
 */
struct ScoreDesc {
  double nu; ///< Learning rate; specified by parameter.
  string scorer; ///< Fixed by algorithm.
  double baseScore; ///< Derived from sampled root.

  /**
     @brief Training constructor:  only learning rate is known.
  */
  ScoreDesc(double nu_ = 0.0) : nu(nu_) {
  }


  /**
     @brief Prediction constructor:  all members known.
   */
  ScoreDesc(const tuple<double, double, string>& valTriple) :
  nu(get<0>(valTriple)),
    scorer(get<2>(valTriple)),
    baseScore(get<1>(valTriple)) {
  }
  
  
  ~ScoreDesc() = default;

  /*
    @brief Builds algorithm-specific scorer for response type.
   */
  unique_ptr<class ForestScorer> makeScorer(const class ResponseReg* response,
					    const class Forest* forest,
					    const class Leaf* leaf,
					    const class PredictReg* predict,
					    vector<double> quantile) const;


  unique_ptr<class ForestScorer> makeScorer(const class ResponseCtg* response,
					    size_t nObs,
					    bool doProb) const;
};

#endif
