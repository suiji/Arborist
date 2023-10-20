// This file is part of deframe.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file rleframe.h

   @brief Run-length encoded representation of data frame.

   @author Mark Seligman
 */

#ifndef DEFRAME_RLEFRAME_H
#define DEFRAME_RLEFRAME_H

#include "rlecresc.h"

/**
   @brief Sorts on row, for reorder.
*/
template<typename valType>
bool RLECompareRow (const RLEVal<valType>& a, const RLEVal<valType>& b) {
  return (a.row < b.row);
}


/**
   @brief Completed form, constructed from front end representation.
 */
struct RLEFrame {
  const size_t nObs;
  const vector<unsigned int> factorTop; ///> top factor index / 0.
  const size_t noRank; ///> Inattainable rank index.
  vector<vector<RLEVal<szType>>> rlePred;
  vector<vector<double>> numRanked;
  vector<vector<unsigned int>> facRanked;
  vector<unsigned int> blockIdx; ///> position of value in block.

  /**
     @brief Constructor from unpacked representation.
   */
  RLEFrame(size_t nObs_,
	   const vector<unsigned int>& factorTop_,
	   const vector<size_t>& runVal,
	   const vector<size_t>& runLength,
	   const vector<size_t>& runObs,
	   const vector<size_t>& rleHeight_,
	   const vector<double>& numVal_,
	   const vector<size_t>& numHeight_,
	   const vector<unsigned int>& facVal_,
	   const vector<size_t>& facHeight_);


  /**
     @brief Builds the per-predictor vectors of run-length encodings.
   */
  static vector<vector<RLEVal<szType>>> packRLE(const vector<size_t>& rleHeight,
						const vector<size_t>& runVal,
						const vector<size_t>& runRow,
						const vector<size_t>& runLength);


  /**
     @brief Row count getter.
   */
  const auto getNRow() const {
    return nObs;
  }

  /**
     @brief Predictor count getter.
   */
  const auto getNPred() const {
    return rlePred.size();
  }


  /**
     @return position of first numerical predictor.
   */
  const auto getNumFirst() const {
    return 0;
  }

  
  /**
     @brief Numeric predictor count getter.
   */
  const auto getNPredNum() const {
    return numRanked.size();
  }


  /**
     @brief Numeric predictor count getter.
   */
  const auto getNPredFac() const {
    return facRanked.size();
  }


  unsigned int getBlockIdx(unsigned int predIdx) const {
    return blockIdx[predIdx];
  }


  unsigned int getFactorTop(unsigned int predIdx) const {
    return factorTop[predIdx];
  }
  

  const vector<RLEVal<szType>>& getRLE(unsigned int predIdx) const {
    return rlePred[predIdx];
  }

  
  /**
     @brief Derives # distinct values, including possible NA.
     
     @param predIdx is the predictor index.

     @return (zero-based) rank of rear, plus one.
   */
  size_t getRunCount(unsigned int predIdx) const {
    return rlePred[predIdx].back().val + 1;
  }


  /**
     @brief Reorders the predictor RLE vectors by row.
   */
  void reorderRow();


  /**
     @return rank index of missing data, if any, else noRank.
   */
  size_t findRankMissing(unsigned int predIdx) const;


  vector<RLEVal<szType>> permute(unsigned int predIdx,
				 const vector<size_t>& idxPerm) const;


  /**
     @brief Obtains the predictor rank at a given row.

     @param[in, out] idxTr gives the element referencing row.

     @param obsIdx is the specified row.

     @return rank at the given row, per predictor.
   */
  vector<szType> idxRank(vector<size_t>& idxTr,
			 size_t obsIdx) const {
    vector<szType> rankVec(idxTr.size());
    for (unsigned int predIdx = 0; predIdx < rankVec.size(); predIdx++) {
      if (obsIdx >= rlePred[predIdx][idxTr[predIdx]].getRowEnd()) {
	idxTr[predIdx]++;
      }
      rankVec[predIdx] = rlePred[predIdx][idxTr[predIdx]].val;
    }
    return rankVec;
  }

  void transpose(vector<size_t>& idxTr,
		 size_t obsStart,
		 size_t extent,
		 vector<double>& num,
		 vector<unsigned int>& fac) const {
    for (size_t obsIdx = obsStart; obsIdx != min(nObs, obsStart + extent); obsIdx++) {
      unsigned int numIdx = 0;
      unsigned int facIdx = 0;
      unsigned int predIdx = 0;
      for (auto rank : idxRank(idxTr, obsIdx)) {
	if (factorTop[predIdx] == 0) {
	  num.push_back(numRanked[numIdx++][rank]);
	}
	else {// TODO:  Replace subtraction with (front end)::fac2Rank()
	  fac.push_back(facRanked[facIdx++][rank] - 1);
	}
	predIdx++;
      }
    }
  }
};


#endif
