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

#include <vector>
using namespace std;


/**
   @brief Run length-encoded representation of pre-sorted frame.

   Crescent form.
 */
class RLECresc {
  const size_t nRow;

  // Empty iff no factor-valued predictors.
  vector<unsigned int> cardinality;

  // Error if empty.
  vector<unsigned int> rank;
  vector<unsigned int> row;
  vector<unsigned int> runLength;
  // TODO:  replace separate vectors with single vector 'rle'.
  // vector<RLE<RowRank> > rle;


  // Sparse numerical representation for split interpolation.
  // Empty iff no numerical predictors.
  vector<unsigned int> numOff; // Per-predictor offset within RLE vectors.
  vector<double> numVal; // Rank-indexable predictor values.

  unsigned int numSortSparse(const double feColNum[],
			     const unsigned int feRowStart[],
			     const unsigned int feRunLength[]);

  // Sparse representation imposed by front-end.
  typedef tuple<double, unsigned int, unsigned int> NumRLE;

  /**
     @brief Stores ordered predictor column, entering uncompressed.

     @param rleNum is a sparse representation of the value/row-number pairs.
  */
  void rankNum(const vector<NumRLE> &rleNum);


 public:

  RLECresc(size_t nRow_,
           unsigned int nPredNum,
           unsigned int nPredFac);

  auto getNRow() const {
    return nRow;
  }

  auto getNPredNum() const {
    return numOff.size();
  }

  auto getNPredFac() const {
    return cardinality.size();
  }

  /**
     @brief Accessor for copyable rank vector.
   */
  const vector<unsigned int>& getRank() const {
    return rank;
  }

  /**
     @brief Accessor for copyable row vector.
   */
  const vector<unsigned int>& getRow() const {
    return row;
  }

  /**
     @brief Accessor for copyable run-length vector.
   */
  const vector<unsigned int>& getRunLength() const {
    return runLength;
  }


  /** 
      @brief Accessor for copyable offset vector.
   */
  const vector<unsigned int>& getNumOff() const {
    return numOff;
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
  const unsigned int* rank;
  const unsigned int* row;
  const unsigned int* runLength;
  const unsigned int nPredNum;
  const double* numVal;
  const unsigned int* numOff;

  RLEFrame(size_t nRow_,
           const vector<unsigned int>& cardinality_,
           size_t rlLength_,
           const unsigned int* row_,
           const unsigned int* rank_,
           const unsigned int* runLength_,
           unsigned int nPredNum_,
           const double* numVal_,
           const unsigned int* numOff_);

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

