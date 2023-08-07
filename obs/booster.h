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


#include <numeric>
#include <algorithm>
#include <vector>

/**
   @brief Maintains boosted estimate.
 */
struct Booster {
  static unique_ptr<Booster> booster; ///< Singleton.

  const double nu; ///< Learning rate.
  double baseScore; ///< Initial etimate, assigned from root.
  vector<double> estimate; ///< Accumulated estimate.

  // Non-incremental updates only:
  vector<SampleNux> baseSamples; ///< Cached bagged samples.


  Booster(double (Booster::*)(const class IndexSet&) const,
	  void (Booster::*)(class FrontierScorer*, class SampledObs*, double&),
	  double nu_);

  
  double (Booster::* baseScorer)(const class IndexSet&) const;

  void setBaseScore(const IndexSet& iSet) const {
    (this->*baseScorer)(iSet);
  }
  

  /**
     @brief Passes through to member.
   */
  static void setEstimate(const class SampledObs*);

  
  void (Booster::* updater)(class FrontierScorer*, class SampledObs*, double&);

  
  /**
     @brief Invokes updater.
   */
  static void updateResidual(class FrontierScorer*,
			     class SampledObs* sampledObs,
			     double& bagSum);

  /**
     @brief Invokes pointer-to-member-function.
   */
  void update(class FrontierScorer* frontierScorer,
	      class SampledObs* sampledObs,
	      double& bagSum) {
    (this->*updater)(frontierScorer, sampledObs, bagSum);
  }

  
  //  vector<SampleNux> score(const class SampledObs* sampledObs,
  //			  double& bagSum);


  /**
     @brief Sets the base estimate.
   */
  void baseEstimate(const class SampledObs* sampledObs);
  

  /**
     @brief Stubbed object for no boosting.
   */
  static void makeZero();

  
  /**
     @brief Boosting with L2 loss.
   */
  static void makeL2(double nu);

  /**
     @brief Boosting with log-odds loss.
   */
  static void makeLogOdds(double nu);


  static void deInit();

  /**
     @return true iff a positive learning rate has been specified.
   */
  static bool boosting() {
    return booster->nu > 0.0;
  }


  /**
     @brief Records per-sample scores from trained tree.
   */
  static void updateEstimate(const class PreTree* preTree,
			     const struct SampleMap& sampleMap);


  void scoreSamples(const class PreTree* preTree,
		    const struct SampleMap& sampleMap);


  double zero(const class IndexSet& iRoot) const;
  

  void noUpdate(class FrontierScorer* frontierScorer,
		class SampledObs* sampledObs,
		double& bagSum);

  void updateL2(class FrontierScorer* frontierScorer,
		class SampledObs* sampledObs,
		double&bagSum);


  void updateLogOdds(class FrontierScorer* frontierScorer,
		     class SampledObs* sampledObs,
		     double& bagSum);

  /**
     @brief Logistically transforms vector of log-odds values.

     @return logistically-transformed vector of probabilities.
   */
  static vector<double> logistic(const vector<double>& logOdds);


  /**
     @brief Scales a vector of probabilities by its complement.

     @return vector of scaled probabilities.
   */
  static vector<double> scaleComplement(const vector<double>& p);

  
  double mean(const class IndexSet& iRoot) const;

  double logit(const class IndexSet& iRoot) const;

  /**
     @brief Reports score descriptor back to trainer.
   */
  static struct ScoreDesc getScoreDesc();
};

#endif
