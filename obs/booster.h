// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file booster.h

   @brief Scoring methods for base.

   @author Mark Seligman
 */

#ifndef OBS_BOOSTER_H
#define OBS_BOOSTER_H


#include "typeparam.h"
#include "samplenux.h"
#include "scoredesc.h"

#include <numeric>
#include <algorithm>
#include <vector>

/**
   @brief Maintains boosted estimate.
 */
struct Booster {
  static unique_ptr<Booster> booster; ///< Singleton.

  ScoreDesc scoreDesc; ///< Completes and hands back to trainer.
  vector<double> estimate; ///< Accumulated estimate.

  // Non-incremental updates only:
  vector<SampleNux> baseSamples; ///< Cached bagged samples.


  Booster(double (Booster::*)(const class Response*) const,
	  void (Booster::*)(struct NodeScorer*, class SampledObs*, double&),
	  double nu_);

  
  double (Booster::* baseScorer)(const class Response*) const;


  void setBaseScore(const class Response* response) const {
    (this->*baseScorer)(response);
  }


  /**
     @brief Specifies forest scorer as plurality.
   */
  static void setMean();


  /**
     @brief Specifies forest scorer as plurality.
   */
  static void setPlurality();


  
  /**
     @brief Passes through to member.
   */
  static void setEstimate(const class Sampler* sampler);

  
  void (Booster::* updater)(struct NodeScorer*, class SampledObs*, double&);

  
  /**
     @brief Invokes updater.
   */
  static void updateResidual(struct NodeScorer*,
			     class SampledObs* sampledObs,
			     double& bagSum);

  /**
     @brief Invokes pointer-to-member-function.
   */
  void update(struct NodeScorer* nodeScorer,
	      class SampledObs* sampledObs,
	      double& bagSum) {
    (this->*updater)(nodeScorer, sampledObs, bagSum);
  }

  
  /**
     @brief Sets the base estimate.
   */
  void baseEstimate(const class Sampler* sampler);


  static void init(const string& loss,
		   const string& scorer,
		   double nu);


  static void deInit();

  /**
     @return true iff a positive learning rate has been specified.
   */
  static bool boosting() {
    return booster->scoreDesc.nu > 0.0;
  }


  /**
     @brief Records per-sample scores from trained tree.
   */
  static void updateEstimate(const class SampledObs* sampledObs,
			     const class PreTree* preTree,
			     const struct SampleMap& sampleMap);


  void scoreSamples(const class SampledObs* sampledObs,
		    const class PreTree* preTree,
		    const struct SampleMap& sampleMap);


  double zero(const class Response* response) const;


  void noUpdate(struct NodeScorer* nodeScorer,
		class SampledObs* sampledObs,
		double& bagSum);

  
  void updateL2(struct NodeScorer* nodeScorer,
		class SampledObs* sampledObs,
		double&bagSum);


  void updateLogOdds(struct NodeScorer* nodeScorer,
		     class SampledObs* sampledObs,
		     double& bagSum);


  double mean(const class Response* response) const;
  

  double logit(const class Response* response) const;


  /**
     @brief Reports contents of score descriptor.
   */
  static void listScoreDesc(double& nu,
			    double& baseScore,
			    string& scorer) {
    nu = booster->scoreDesc.nu;
    baseScore = booster->scoreDesc.baseScore;
    scorer = booster->scoreDesc.scorer;
  }
};

#endif
