// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file frontierscorer.h

   @brief Scoring methods for frontier.

   @author Mark Seligman
 */

#ifndef OBS_FRONTIERSCORER_H
#define OBS_FRONTIERSCORER_H


#include "typeparam.h"


#include <vector>
#include <numeric>
#include <algorithm>


struct FrontierScorer {
  vector<double> ctgJitter; ///< Breaks classification ties.
  vector<double> gamma; ///< Per-sample weight, with multiplicity.

  
  double (FrontierScorer::* scorer)(const class SampleMap&,
				    const class IndexSet&) const;

  FrontierScorer(double (FrontierScorer::* scorer_)(const class SampleMap&,
						    const class IndexSet&) const);

  void frontierPreamble(const class Frontier* frontier);
  
  static unique_ptr<FrontierScorer> makeMean();

  static unique_ptr<FrontierScorer> makePlurality();

  static unique_ptr<FrontierScorer> makeLogOdds();

  
  double score(const class SampleMap& smNonterm,
	       const class IndexSet& iSet) const {
    return (this->*scorer)(smNonterm, iSet);
  }


  void setGamma(vector<double> prob) {
    gamma = std::move(prob);
  }
  
  
  /**
     @return mean reponse over node.
   */
  double scoreMean(const class SampleMap& smNonterm,
		   const class IndexSet& iSet) const;


  /**
     @return category with jittered plurality, plus jitter.
   */
  double scorePlurality(const class SampleMap& smNonterm,
			const class IndexSet& iSet) const;


  /**
     @return mean score weighted by per-sample p-q probabilities.
   */
  double scoreLogOdds(const class SampleMap& smNonterm,
		      const class IndexSet& iSet) const;
};

#endif
