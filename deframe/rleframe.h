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
  const vector<PredictorForm> predForm;
  vector<vector<RLEVal<unsigned int>>> rlePred;
  const vector<double> numVal;
  const vector<size_t> numHeight;
  const vector<unsigned int> facVal;
  const vector<size_t> facHeight;

  
  /**
     @brief Constructor from unpacked representation.
   */
  RLEFrame(size_t nRow_,
	   const vector<PredictorForm>& predForm_,
	   const vector<size_t>& runVal,
	   const vector<size_t>& runLenght,
	   const vector<size_t>& runRow,
	   const vector<size_t>& rleHeight_,
	   const vector<double>& numVal_,
	   const vector<size_t>& numHeight_,
	   const vector<unsigned int>& facVal_,
	   const vector<size_t>& facHeight_);

  
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
    return numHeight.size();
  }


  const vector<RLEVal<unsigned int>>& getRLE(unsigned int predIdx) const {
    return rlePred[predIdx];
  }

  
  vector<RLEVal<unsigned int>> permute(unsigned int predIdx,
	       const vector<size_t>& idxPerm) const;
};

#endif
