// This file is part of deframe.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file rlecresc.h

   @brief Run-length encoded representation of data frame.

   @author Mark Seligman
 */

#ifndef DEFRAME_RLECRESC_H
#define DEFRAME_RLECRESC_H

#include "rle.h"
#include "valrank.h"

#include <cstdint>
#include <vector>

using namespace std;


/**
   @brief Sorts on value, then rows, for stability.
 */
template<typename valType>
bool RLECompare (const RLEVal<valType>& a,
		 const RLEVal<valType>& b) {
  //   N.B:  extraneous parentheses avoid parser error in older g++.
  return (a.val < b.val) || (areEqual(a.val, b.val) && ((a.row) < b.row));
}


template<>
inline bool RLECompare(const RLEVal<double>& a,
		       const RLEVal<double>& b) {
  return (a.val < b.val) || (areEqual(a.val, b.val) && ((a.row) < b.row)) || (!isnan(a.val) && isnan(b.val));
}


typedef size_t szType; // Size type sufficient for observations.


/**
   @brief Run length-encoded representation of pre-sorted frame.

   Crescent form.
 */
class RLECresc {
  const szType nRow; ///> # observations.

  vector<unsigned int> topIdx; ///> highest FE index or 0 if numeric.
  vector<unsigned int> typedIdx; // Maps index to typed offset.

  // Encodes observations as run characteristics, not values.
  // Error if empty.
  vector<vector<RLEVal<szType>>> rle;
  vector<vector<unsigned int>> valFac;
  vector<vector<double>> valNum;
  unsigned int nFactor; // Count of factor predictors.
  unsigned int nNumeric; // Count of numeric predictors.

  /**
     @brief Emits a run-length encoding of a sorted list.

     @param runValue is a stably-sorted vector of values and ranks.

     @param[out] runValue outputs unique values in sorted order.
   */
  template<typename obsType>
  void encode(const RankedObs<obsType>& rankedObs,
	      vector<obsType>& runValue,
	      vector<RLEVal<szType>>& rlePred) {
    size_t rowNext = nRow; // Inattainable row number.

    obsType valPrev = rankedObs.getVal(0); // Ensures intial rle pushed at first iteration.
    runValue.push_back(valPrev); // Ensures initial value pushed at first iteration.
    for (size_t idx = 0; idx < nRow; idx++) {
      auto rowThis = rankedObs.getRow(idx);
      obsType valThis = rankedObs.getVal(idx);
      if (!areEqual(valThis, valPrev)) {
	runValue.push_back(valThis);
	rlePred.emplace_back(RLEVal<szType>(rankedObs.getRank(idx), rowThis));
      }
      else if (rowThis != rowNext) {
	rlePred.emplace_back(RLEVal<szType>(rankedObs.getRank(idx), rowThis));
      }
      else {
	rlePred.back().extent++;
      }
      valPrev = valThis;
      rowNext = rowThis + 1;
    }
  }

  
  /**
     @brief Presorts runlength-encoded numerical block supplied by front end.

     @param feVal[] is a vector of valType values.

     @param feRowStart[] maps row indices to offset within value vector.

     @param feRunLength[] is length of each run of values.
   */
  template<typename valType>
  vector<vector<valType>> encodeSparse(unsigned int nPredType,
				       const vector<valType>& feVal,
				       const vector<size_t>& feRowStart,
				       const vector<size_t>& feRunLength) {
    vector<vector<valType>> val(nPredType);
    size_t colOff = 0;
    unsigned int predIdx = 0;
    for (auto & runValue : val) {
      colOff += sortSparse(runValue, predIdx++, &feVal[colOff], &feRowStart[colOff], &feRunLength[colOff]);
    }

    return val;
  }


  template<typename valType>
  size_t sortSparse(vector<valType>& runValue,
		    unsigned int predIdx,
		    const double feCol[],
		    const size_t feRowStart[],
		    const size_t feRunLength[]) {
    vector<RLEVal<valType> > rleVal;
    size_t rleIdx = 0;
    for (size_t rowTot = 0; rowTot < nRow; rowTot += feRunLength[rleIdx++]) {
      rleVal.emplace_back(RLEVal<valType>(feCol[rleIdx], feRowStart[rleIdx], feRunLength[rleIdx]));
    }
    // Postcondition:  rleNum.size() == caller's vector length.

    sort(rleVal.begin(), rleVal.end(), RLECompare<valType>);
    encodeSparse(runValue, rleVal, rle[predIdx]);

    return rleVal.size();
  }


  /**
     @brief Stores ordered predictor column, entering uncompressed.

     @param rleVal is a sparse representation of the value/row-number pairs.
  */
  template<typename valType>
  void encodeSparse(vector<valType>& runValue,
		    const vector<RLEVal<valType>>& rleVal,
		    vector<RLEVal<szType>>& rlePred) {
    size_t rowNext = nRow; // Inattainable row number.
    size_t rk = 0;
    runValue.push_back(rleVal[0].val);
    for (auto elt : rleVal) {
      bool tied = areEqual(elt.val, runValue.back());
      if (tied && elt.row == rowNext) { // Run continues.
	rlePred.back().extent += elt.extent;
      }
      else { // New RLE, rank entries regardless whether tied.
	if (!tied) {
	  rk++;
	  runValue.push_back(elt.val);
	}
	rlePred.emplace_back(RLEVal<szType>(rk, elt.row, elt.extent));
      }
      rowNext = rlePred.back().row + rlePred.back().extent;
    }
  }


public:

  RLECresc(size_t nRow_,
	   unsigned int nPred);


  auto getNRow() const {
    return nRow;
  }

  
  /**
     @brief Computes unit size for cross-compatibility of serialization.
   */
  static constexpr size_t unitSize() {
    return sizeof(RLEVal<szType>);
  }


  /**
     @param topIdx is the highest factor index, excluding NA proxy.

     'topIdx' records the factor encoding employed by the front end,
     irrespective of whether the level indices are zero- or one-based.
   */
  void setFactor(unsigned int predIdx,
		 unsigned int topIdx) {
    if (topIdx > 0) {
      typedIdx[predIdx] = nFactor++;
    }
    else {
      typedIdx[predIdx] = nNumeric++;
    }
    this->topIdx[predIdx] = topIdx;
  }

  unsigned int getTypedIdx(unsigned int predIdx,
			   bool& isFactor) const {
    isFactor = topIdx[predIdx] > 0;
    return typedIdx[predIdx];
  }
  

  auto getNFactor() const {
    return nFactor;
  }

  
  auto getNNumeric() const {
    return nNumeric;
  }

  
  vector<unsigned int> dumpTopIdx() const {
    return topIdx;
  }

  
  auto getTypedIdx(unsigned int predIdx) const {
    return typedIdx[predIdx];
  }


  const vector<vector<unsigned int>>& getValFac() const {
    return valFac;
  }


  const vector<vector<double>>& getValNum() const {
    return valNum;
  }
  

  /**
     @brief Accessor for copyable height vector.
   */
  const vector<size_t> getHeight() const {
    vector<size_t> rleHeight(rle.size());
    unsigned int predIdx = 0;
    size_t totHeight = 0;
    for (auto & height : rleHeight) {
      totHeight += rle[predIdx++].size();
      height = totHeight;
    }
    
    return rleHeight;
  }


  /**
     @brief Encodes a frame consisting of factors and/or numeric values.

     @param colBase is the base address of each column (predictor).
   */
  void encodeFrame(const vector<void*>& colBase);


  /**
     @brief Encodes entire frame from sparse numeric specification.
   */
  void encodeFrameNum(const vector<double>&  feVal,
		      const vector<size_t>&  feRowStart,
		      const vector<size_t>&  feRunLength);


  /**
     @brief Encodes entire frame from dense numeric block.
   */
  void encodeFrameNum(const double*  feVal);
  

  /**
     @brief As above, but encodes factor-valued frame.
   */
  void encodeFrameFac(const uint32_t* feVal);


  void dump(vector<size_t>& valOut,
	    vector<size_t>& lengthOut,
	    vector<size_t>& rowOut) const;
  

  /**
     @brief Dumps packed structures as raw bytes.
   */
  void dumpRaw(unsigned char rleRaw[]) const;


  /**
     @brief Sorts and run-encodes a contiguous set of predictor values.

     @param val points to the base of the column.

     @param[out] valOut tabulates the predictor values.

     @param[out] rleVal encodes the run-length elements.
   */
  template<typename valType>
  void encodeColumn(const valType val[],
		    vector<valType>& valOut,
		    vector<RLEVal<szType>>& rleVal) {
    encode(RankedObs<valType>(val, nRow), valOut, rleVal);
  }
};
#endif

