// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file rleframe.h

   @brief Run-length encoded representation of data frame.

   @author Mark Seligman
 */

#ifndef FRAMEMAP_RLEFRAME_H
#define FRAMEMAP_RLEFRAME_H

#include "rle.h"
#include "rowrank.h"
#include "valrank.h"

#include <vector>
using namespace std;

/**
  @brief Sparse representation imposed by front-end.
*/
template<typename valType>
struct RLEVal {
  valType val;
  unsigned int row;
  unsigned int runLength;

  RLEVal(valType val_,
         unsigned int row_,
         unsigned int runLength_) : val(val_),
                                    row(row_),
                                    runLength(runLength_) {
  }
};


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

  // Error if empty.
  vector<RLEVal<unsigned int> > rle;

  // Sparse numerical representation for split interpolation.
  // Empty iff no numerical predictors.
  vector<unsigned int> valOff; // Per-predictor offset within RLE vectors.
  vector<double> numVal; // Rank-indexable predictor values.

  unsigned int numSortSparse(const double feColNum[],
			     const unsigned int feRowStart[],
			     const unsigned int feRunLength[]);

  //  typedef tuple<double, unsigned int, unsigned int> NumRLE;

  /**
     @brief Stores ordered predictor column, entering uncompressed.

     @param rleNum is a sparse representation of the value/row-number pairs.
  */
  void encode(const vector<RLEVal<double> > &rleNum);


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
  const vector<unsigned int>& getValOff() const {
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

  
  void dumpRLE(unsigned char rleRaw[]) const;


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


/**
   @brief Completed form, constructed from front end representation.
 */
struct RLEFrame {
  const size_t nRow;
  const vector<unsigned int> cardinality;
  const size_t rleLength;
  const RLEVal<unsigned int>* rle;
  const unsigned int nPredNum;
  const double* numVal;
  const unsigned int* valOff;

  RLEFrame(size_t nRow_,
           const vector<unsigned int>& cardinality_,
           size_t rlLength_,
           const RLEVal<unsigned int>* rle_,
           unsigned int nPredNum_,
           const double* numVal_,
           const unsigned int* valOff_);

  /**
     @brief Row count getter.
   */
  const auto getNRow() const {
    return nRow;
  }

  /**
     @brief Predictor count getter.
   */
  const auto getNPred() const {
    return nPredNum + cardinality.size();
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
    return nPredNum;
  }

  
  const vector<unsigned int>& getCardinality() const {
    return cardinality;
  }
};


#endif

