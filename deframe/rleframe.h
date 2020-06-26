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

#ifndef DEFRAME_RLEFRAME_H
#define DEFRAME_RLEFRAME_H

#include "rlecresc.h"


/**
   @brief Completed form, constructed from front end representation.
 */
struct RLEFrame {
  const size_t nRow;
  const vector<unsigned int> cardinality;
  const unsigned int nPred;
  vector<vector<RLEVal<unsigned int>>> rlePred;
  const unsigned int nPredNum;
  const vector<double> numVal;
  const vector<size_t> numOff;


  /**
     @brief Constructor from packed representation.
   */
  RLEFrame(size_t nRow_,
	   const vector<unsigned int>& cardinality_,
	   const RLEVal<unsigned int>* rle_,
	   const vector<size_t>& rleHeight_,
	   const vector<double>& numVal_,
	   const vector<size_t>& numOff_);


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
    return nPred;
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


  const vector<RLEVal<unsigned int>>& getRLE(unsigned int predIdx) const {
    return rlePred[predIdx];
  }

  
  vector<RLEVal<unsigned int>> permute(unsigned int predIdx,
	       const vector<size_t>& idxPerm) const;
};

#endif
