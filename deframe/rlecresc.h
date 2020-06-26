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
   @brief Run length-encoded representation of pre-sorted frame.

   Crescent form.
 */
class RLECresc {
  const size_t nRow;

  // Empty iff no factor-valued predictors.
  vector<unsigned int> cardinality;

  // Encodes run characteristics, not values.
  // Error if empty.
  vector<RLEVal<unsigned int> > rle;

  // Per-predictor (blocked) cumulative heights.
  vector<size_t> rleHeight;
  
  // Sparse numerical representation for split interpolation.
  // Empty iff no numerical predictors.
  vector<size_t> valOff; // Per-predictor offset within RLE vectors.
  vector<double> numVal; // Rank-indexable predictor values.

  size_t numSortSparse(const double feColNum[],
		       const unsigned int feRowStart[],
		       const unsigned int feRunLength[]);

  //  typedef tuple<double, unsigned int, unsigned int> NumRLE;

  /**
     @brief Stores ordered predictor column, entering uncompressed.

     @param rleNum is a sparse representation of the value/row-number pairs.
  */
  void encode(const vector<RLEVal<double> >& rleNum);


  /**
     @brief Emits a run-length encoding of a sorted list.

     @param vr is a stable sorted vector with ranks.

     @param[out] val outputs values associated with the runs.

     @param valUnique is true iff only unique values are to be output.
   */
  template<typename tn>
  void encode(ValRank<tn>& vr, vector<tn>& val, bool valUnique = true);

 public:

  RLECresc(size_t nRow_,
           unsigned int nPredNum,
           unsigned int nPredFac);

  auto getNRow() const {
    return nRow;
  }

  auto getNPredNum() const {
    return valOff.size();
  }

  auto getNPredFac() const {
    return cardinality.size();
  }

  /**
     @brief Computes unit size for cross-compatibility of serialization.
   */
  static constexpr size_t unitSize() {
    return sizeof(RLEVal<unsigned int>);
  }


  /** 
      @brief Accessor for copyable offset vector.
   */
  const vector<size_t>& getNumOff() const {
    return valOff;
  }

  /**
     @brief Accessor for copyable numerical value vector.
   */
  const vector<double>& getNumVal() const {
    return numVal;
  }
  

  /**
     @brief Accessor for copyable cardinality vector.
   */
  const vector<unsigned int>& getCardinality() const {
    return cardinality;
  }


  size_t getRLEBytes() const {
    return sizeof(RLEVal<unsigned int>) * rle.size();
  }

  /**
     @brief Accessor for copyable height vector.
   */
  const vector<size_t>& getHeight() const {
    return rleHeight;
  }


  void dump(vector<size_t>& valOut,
	    vector<size_t>& lengthOut,
	    vector<size_t>& rowOut) const;
  

  /**
     @brief Dumps packed structures as raw bytes.
   */
  void dumpRaw(unsigned char rleRaw[]) const;


  /**
     @brief Presorts runlength-encoded numerical block supplied by front end.

     @param feValNum[] is a vector of numerical values.

     @param feRowStart[] maps row indices to offset within value vector.

     @param feRunLength[] is length of each run of values.
   */
  void numSparse(const double feValNum[],
		 const unsigned int feRowStart[],
		 const unsigned int feRunLength[]);


  /**
     @brief Presorts dense numerical block supplied by front end.

     @param feNum references a block of factor-valued data.
   */
  void numDense(const double feNum[]);

  
  /**
     @brief Presorts factors and stores as rank-ordered run-length encoding.

     Assumes 0-justification ensured by bridge.

     @param feFac references a block of factor-valued data.

     Final "rank" values are the internal factor codes and may contain
     gaps.  A dense numbering scheme would entail backmapping at LH bit
     assignment following splitting (q.v.):  prediction and training
     must be able to reconcile separately-assigned factor levels.
  */ 
  void facDense(const unsigned int feFac[]);
};
#endif

