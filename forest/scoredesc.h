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

#include "prediction.h"
#include "typeparam.h"

#include <vector>
#include <string>


class Predict;
class Sampler;

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


  tuple<double, double, string> getTuple() {
    return tuple<double, double, string>(nu, baseScore, scorer);
  }


  /*
    @brief Builds algorithm-specific scorer for response type.
   */
  unique_ptr<ForestPredictionReg> makePredictionReg(const Predict* predict,
						    const Sampler* sampler,
						    bool reportAuxiliary) const;


  unique_ptr<ForestPredictionCtg> makePredictionCtg(const Predict* predict,
						    const Sampler* sampler,
						    bool reportAuxiliary) const;
};

#endif
