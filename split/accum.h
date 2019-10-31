/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#ifndef SPLIT_ACCUM_H
#define SPLIT_ACCUM_H

/**
   @file accum.h

   @brief Base accumulator classes for cut-based (numeric) splitting workspaces.

   @author Mark Seligman

 */

#include "typeparam.h"


/**
   @brief Persistent workspace for computing optimal split.

   Cells having implicit dense blobs are split in separate sections,
   calling for a re-entrant data structure to cache intermediate state.
   Accum is tailored for right-to-left index traversal.
 */
class Accum {
protected:
  const IndexT sCount; // Running sample count along node.
  const double sum; // Running response along node.
  const IndexT rankDense; // Rank of dense value, if any.
  IndexT sCountL; // Running sum of trial LHS sample counts.
  double sumL; // Running sum of trial LHS response.
  IndexT cutDense; // Rightmost position beyond implicit blob, if any.
  
  // Read locally but initialized, and possibly reset, externally.
  IndexT sCountThis; // Current sample count.
  FltVal ySum; // Current response value.


  /**
     @brief Updates split anywhere left of a residual, if any.
   */
  inline void trialRight(double infoTrial,
			 IndexT idx,
			 IndexT rkThis,
			 IndexT rkRight) {
    if (infoTrial > info) {
      info = infoTrial;
      lhSCount = sCountL;
      rankRH = rkRight;
      rankLH = rkThis;
      rhMin = rkRight == rankDense ? cutDense : idx + 1;
    }
  }

  /**
     @brief Updates split just to the right of a residual.
   */
  inline void splitResidual(double infoTrial,
			   IndexT rkRight) {
    if (infoTrial > info) {
      info = infoTrial;
      lhSCount = sCountL;
      rankRH = rkRight;
      rankLH = rankDense;
      rhMin = cutDense;
    }
  }
  
public:
  // Revised at each new local maximum of 'info':
  double info; // Information high watermark.  Precipitates split iff > 0.0.
  IndexT lhSCount; // Sample count of split LHS:  > 0.
  IndexT rankRH; // Maximum rank characterizing split.
  IndexT rankLH; // Minimum rank charactersizing split.
  IndexT rhMin; // Min RH index, possibly out of bounds:  [0, idxEnd+1].
  
  Accum(const class SplitNux* cand,
        IndexT rankDense_);

  
  ~Accum() {
  }

  
  /**
     @brief Creates a residual summarizing implicit splitting state.

     @param cand is the splitting candidate.

     @param spn is the splitting data set.
     
     @return new residual based on the current splitting data set.
   */
  unique_ptr<struct Residual> makeResidual(const class SplitNux* cand,
                                          const class SampleRank spn[]);


  IndexT lhImplicit(const class SplitNux* cand) const;
};

#endif

