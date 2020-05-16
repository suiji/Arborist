/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#ifndef SPLIT_CUTACCUM_H
#define SPLIT_CUTACCUM_H

/**
   @file cutaccum.h

   @brief Base accumulator classes for cut-based (numeric) splitting workspaces.

   @author Mark Seligman

 */

#include "typeparam.h"
#include "accum.h"

#include <vector>


/**
   @brief Persistent workspace for computing optimal split.

   Cells having implicit dense blobs are split in separate sections,
   calling for a re-entrant data structure to cache intermediate state.
   Accum is tailored for right-to-left index traversal.
 */
class CutAccum : public Accum {
protected:
  const class SplitFrontier* splitFrontier;
  IndexT sCount; // Running sum of trial LHS sample counts.
  double sum; // Running sum of trial LHS response.
  IndexT cutDense; // Rightmost position beyond implicit blob, if any.
  
  // Read locally but initialized, and possibly reset, externally.
  IndexT sCountThis; // Current sample count.
  FltVal ySumThis; // Current response value.


  /**
     @brief Updates split anywhere left of a residual, if any.
   */
  inline void trialRight(double infoTrial,
			 IndexT idxLeft,
			 IndexT rkThis,
			 IndexT rkRight) {
    if (infoTrial > info) {
      info = infoTrial;
      lhSCount = sCount;
      lhSum = sum;
      rankRH = rkRight;
      rankLH = rkThis;
      this->idxLeft = idxLeft;
      idxRight = rkRight == rankDense ? cutDense : idxLeft + 1;
    }
  }


  /**
     @brief As above, but with distinct index bounds.
   */
  void trialSplit(double infoTrial,
		  IndexT idxLeft,
		  IndexT idxRight);

  
  /**
     @brief Updates split just to the right of a residual.
   */
  inline void splitResidual(double infoTrial,
			   IndexT rkRight) {
    if (infoTrial > info) {
      info = infoTrial;
      lhSCount = sCount;
      lhSum = sum;
      rankRH = rkRight;
      rankLH = rankDense;
      idxRight = cutDense;
    }
  }
  
public:
  // Revised at each new local maximum of 'info':
  IndexT lhSCount; // Sample count of split LHS:  > 0.
  double lhSum; // Sum of responses over LHS.
  IndexT rankRH; // Maximum rank characterizing split.
  IndexT rankLH; // Minimum rank charactersizing split.
  IndexT idxLeft; // sup left index.  Out of bounds (idxEnd + 1) iff left is dense.
  IndexT idxRight; // inf right index.  Out of bounds (idxEnd + 1) iff right is dense.

  /**
     @param cand encapsulates candidate splitting parameters.

     @param splitFrontier looks up dense rank.
   */
  CutAccum(const class SplitNux* cand,
	   const class SplitFrontier* splitFrontier);

  ~CutAccum() {
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


  double interpolateRank(double splitQuant) const;
};


/**
   @brief Minimal informatoin needed to reconstruct cut.
 */
struct CutSig {
  // In CART-like implementations, idxLeft and idxRight are adjacent.
  IndexT idxLeft; // sup of left SampleRank indices.
  IndexT idxRight;  // inf of right SampleRank indices.
  IndexT implicitTrue; // # implicit SampleRank indices associated with true sense.
  double quantRank; // Interpolated cut rank.
  bool cutLeft; // True iff cut encoded by left portion.

  CutSig(const IndexRange& idxRange) :
    idxRight(idxRange.getEnd() - 1),
    cutLeft(true) { // Default.
  }

  CutSig() :
    cutLeft(true) {
  }
};




class CutSet {
  vector<CutSig> cutSig;

public:
  CutSet();


  CutSig getCut(IndexT accumIdx) const {
    return cutSig[accumIdx];
  }


  /**
     @brief Same as above, but looks up from nux accum index.
   */
  CutSig getCut(const SplitNux& nux) const;

  
  void setCut(IndexT accumIdx, const CutSig& sig) {
    cutSig[accumIdx] = sig;
  }
  
  
  IndexT addCut(const class SplitNux* cand);

  
  IndexT getAccumCount() const {
    return cutSig.size();
  }
  

  void write(const class SplitNux* nux,
	     const CutAccum* accum);

  
  /**
     @return true iff cut associated with split has left sense.
   */
  bool leftCut(const class SplitNux* nux) const;


  /**
     @brief Sets the sense of a given cut.
   */
  void setCutSense(IndexT cutIdx,
		   bool sense);

  double getQuantRank(const class SplitNux* nux) const;


  IndexT getIdxRight(const class SplitNux* nux) const;

  
  IndexT getIdxLeft(const class SplitNux* nux) const;

  
  IndexT getImplicitTrue(const class SplitNux* nux) const;
};


#endif

