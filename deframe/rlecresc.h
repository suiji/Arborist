// This file is part of ArboristCore.

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

#include <vector>
using namespace std;

/**
   @brief Sorts on value, then rows, for stability.

   N.B:  extraneous parentheses work around parser error in older g++.
 */
template<typename valType>
bool RLECompare (const RLEVal<valType> &a, const RLEVal<valType>& b) {
  return (a.val < b.val) || ((a.val == b.val) && ((a.row) < b.row));
}


/**
   @brief Sorts on row, for reorder.
 */
template<typename valType>
bool RLECompareRow (const RLEVal<valType> &a, const RLEVal<valType>& b) {
  return (a.row < b.row);
}


/**
   @brief Run length-encoded representation of pre-sorted frame.

   Crescent form.
 */
class RLECresc {
  const size_t nRow;

  vector<PredictorForm> predForm; // Maps predictor index to its form.

  vector<unsigned int> typedIdx; // Maps index to typed offset.

  // Encodes observations as run characteristics, not values.
  // Error if empty.
  vector<vector<RLEVal<unsigned int>>> rle;


  vector<vector<unsigned int>> valFac;


  vector<vector<double>> valNum;


  unsigned int nFactor; // Count of factor predictors.

  unsigned int nNumeric; // Count of numeric predictors.

  
  /**
     @brief Emits a run-length encoding of a sorted list.

     @param valPred is a stably-sorted vector of values and ranks.

     @param[out] val outputs unique values in sorted order.
   */
  template<typename tn>
  void encode(const ValRank<tn>& vr, vector<tn>& valPred, vector<RLEVal<unsigned int>>& rlePred) {
    size_t rowNext = nRow; // Inattainable row number.

    tn valPrev = vr.getVal(0); // Ensures intial rle pushed at first iteration.
    valPred.push_back(valPrev); // Ensures initial value pushed at first iteration.
    for (size_t idx = 0; idx < nRow; idx++) {
      auto rowThis = vr.getRow(idx);
      auto valThis = vr.getVal(idx);
      if (valThis != valPrev) {
	valPred.push_back(valThis);
	rlePred.emplace_back(RLEVal<unsigned int>(vr.getRank(idx), rowThis));
      }
      else if (rowThis != rowNext) {
	rlePred.emplace_back(RLEVal<unsigned int>(vr.getRank(idx), rowThis));
      }
      else {
	rlePred.back().extent++;
      }
      valPrev = valThis;
      rowNext = rowThis + 1;
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
    return sizeof(RLEVal<unsigned int>);
  }


  void setFactor(unsigned int predIdx, bool isFactor) {
    if (isFactor) {
      predForm[predIdx] = PredictorForm::factor;
      typedIdx[predIdx] = nFactor++;
    }
    else {
      predForm[predIdx] = PredictorForm::numeric;
      typedIdx[predIdx] = nNumeric++;
    }
  }

  unsigned int getTypedIdx(unsigned int predIdx,
			   bool& isFactor) const {
    isFactor = predForm[predIdx] == PredictorForm::factor;
    return typedIdx[predIdx];
  }
  

  auto getNFactor() const {
    return nFactor;
  }

  
  auto getNNumeric() const {
    return nNumeric;
  }

  
  vector<unsigned int> dumpPredForm() const {
    vector<unsigned int> predFormOut(predForm.size());
    unsigned int i = 0;
    for (auto pf : predForm) {
      predFormOut[i++] = static_cast<unsigned int>(pf);
    }
    return predFormOut;
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
  

  void dump(vector<size_t>& valOut,
	    vector<size_t>& lengthOut,
	    vector<size_t>& rowOut) const;
  

  /**
     @brief Dumps packed structures as raw bytes.
   */
  void dumpRaw(unsigned char rleRaw[]) const;


  /**
     @brief Presorts runlength-encoded numerical block supplied by front end.

     @param feVal[] is a vector of valType values.

     @param feRowStart[] maps row indices to offset within value vector.

     @param feRunLength[] is length of each run of values.
   */
  template<typename valType>
  vector<vector<valType>> encodeSparse(unsigned int nPredType,
				      const vector<valType>&  feVal,
				      const vector<size_t>&  feRowStart,
				      const vector<size_t>&  feRunLength) {
    vector<vector<valType>> val(nPredType);
    size_t colOff = 0;
    unsigned int predIdx = 0;
    for (auto & valPred : val) {
      colOff += sortSparse(valPred, predIdx++, &feVal[colOff], &feRowStart[colOff], &feRunLength[colOff]);
    }
    return val;
  }


  template<typename valType>
  size_t sortSparse(vector<valType>& valPred,
		    unsigned int predIdx,
		    const double feCol[],
		    const size_t feRowStart[],
		    const size_t feRunLength[]) {
    vector<RLEVal<valType> > rleVal;
    size_t rleIdx = 0;
    for (size_t rowTot = 0; rowTot < nRow; rowTot += feRunLength[rleIdx++]) {
      rleVal.emplace_back(RLEVal<double>(feCol[rleIdx], feRowStart[rleIdx], feRunLength[rleIdx]));
    }
    // Postcondition:  rleNum.size() == caller's vector length.

    sort(rleVal.begin(), rleVal.end(), RLECompare<valType>);
    encode(valPred, rleVal, rle[predIdx]);

    return rleVal.size();
  }

  
  /**
     @brief Stores ordered predictor column, entering uncompressed.

     @param rleVal is a sparse representation of the value/row-number pairs.
  */
  template<typename valType>
  void encode(vector<valType>& valPred,
	      const vector<RLEVal<valType> >& rleVal,
	      vector<RLEVal<unsigned int>>& rlePred) {
    size_t rowNext = nRow; // Inattainable row number.
    size_t rk = 0;
    valPred.push_back(rleVal[0].val);
    for (auto elt : rleVal) {
      valType valThis = elt.val;
      auto rowThis = elt.row;
      auto runCount = elt.extent;
      if (valThis == valPred.back() && rowThis == rowNext) { // Run continues.
	rlePred.back().extent += runCount;
      }
      else { // New RLE, rank entries regardless whether tied.
	if (valThis != valPred.back()) {
	  rk++;
	  valPred.push_back(valThis);
	}
	rlePred.emplace_back(RLEVal<unsigned int>(rk, rowThis, runCount));
      }
      rowNext = rlePred.back().row + rlePred.back().extent;
    }
  }


  /**
     @brief Sorts and run-encodes a contiguous column of values.

     @param val points to the base of the column.

     @param[out] valOut is the vector of encoded values.

     @param predIdx is the predictor (column) index.
   */
  template<typename valType>
  void encodeColumn(const valType* val,
		    vector<valType>& valOut,
		    unsigned int predIdx) {
    ValRank<valType> valRank(&val[0], nRow);
    encode(valRank, valOut, rle[predIdx]);
  }
};
#endif

