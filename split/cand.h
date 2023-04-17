// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#ifndef SPLIT_CAND_H
#define SPLIT_CAND_H

/**
   @file cand.h

   @brief Manages generic splitting candidate selection.

   @author Mark Seligman
 */

#include "typeparam.h"
#include "splitcoord.h"

#include <vector>

/**
   @brief Minimal information needed to preschedule a splitting candidate.
 */

/**
   @brief Minimal information needed to define a splitting pre=candidate.
 */
struct PreCand {
  SplitCoord coord;
  uint32_t randVal; // Arbiter for tie-breaking and the like.
  
  /**
     @brief ObsCell component initialized at construction, StagedCell at (re)staging.
   */
  PreCand(const SplitCoord& coord_,
	  uint32_t randVal_) :
    coord(coord_),
    randVal(randVal_) {
  }

  
  PredictorT getNodeIdx() const {
    return coord.nodeIdx;
  }
};


struct Cand {
  const IndexT nSplit;
  const PredictorT nPred;

  vector<vector<PreCand>> preCand;

  Cand(const class InterLevel* interLevel);
  

  void precandidates(const class Frontier* frontier,
		     class InterLevel* interLevel);


  /**
     @brief Accepts all eligible pairs as precandidates.
   */
  void candidateCartesian(const class Frontier* frontier,
			  class InterLevel* interLevel);


  /**
     @brief Accepts precandidates using Bernoulli sampling.
   */
  void candidateBernoulli(const class Frontier* frontier,
			  class InterLevel* interLevel,
			  const vector<double>& predProb);

  /**
     @brief Samples fixed number of precandidates without replacement.
   */
  void candidateFixed(const class Frontier* frontier,
		      class InterLevel* interLevel,
		      PredictorT predFixed);


  /**
     @return flattened vector of all staged candidates.
   */
  vector<class SplitNux> stagedSimple(const class InterLevel* interLevel,
				      class SplitFrontier* splitFrontier) const;


  /**
     @return vector of per-node vectors of staged candidates.
   */
  vector<vector<class SplitNux>> stagedCompound(const class InterLevel* interLevel,
						class SplitFrontier* splitFrontier) const;


  /**
     @brief Extracts the 32 lowest-order mantissa bits of a double-valued
     random variate.

     The double-valued variants passed are used by the caller to arbitrate
     variable sampling and are unlikely to rely on more than the first
     few mantissa bits.  Hence using the low-order bits to arbitrate other
     choices is unlikely to introduce spurious correlations.
   */
  inline static unsigned int getRandLow(double rVal) {
    union { double d; uint32_t ui[2]; } u = {rVal};
    
    return u.ui[0];
  }
};

#endif
