// This file is part of deframe.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file localframe.h

   @brief Decompressed sections of an RLEFrame.

   @author Mark Seligman
 */

#ifndef DEFRAME_LOCALFRAME_H
#define DEFRAME_LOCALFRAME_H

#include "typeparam.h"

#include <vector>

using namespace std;


/**
   @brief Transposed section of an RLEFrame.
 */
class PredictFrame {
  PredictorT nPredNum; ///< # numeric predictors in progenitor.
  PredictorT nPredFac; ///< # factor " " .
  size_t baseObs; ///< Position of frame within progenitor.
  vector<size_t> idxTr; ///< Per-predictor transposition state.

public:
  vector<double> num;
  vector<CtgT> fac;

  PredictFrame(const class RLEFrame* frame);


  void transpose(const class RLEFrame* frame,
		 size_t obsStart,
		 size_t obsExtent);


  /**
     @return # numeric predictors.

     Deprecate when Forest no longer requires.
   */
  unsigned int getNPredNum() const {
    return nPredNum;
  }


  /**
     @brief Computes block-relative position for a predictor.

     @param[out] thisIsFactor outputs true iff predictor is factor-valued.

     @return block-relative index.
   */
  inline unsigned int getIdx(PredictorT predIdx, bool& predIsFactor) const {
    predIsFactor = isFactor(predIdx);
    return predIsFactor ? predIdx - nPredNum : predIdx;
  }

  
  inline bool isFactor(PredictorT predIdx) const {
    return predIdx >= nPredNum;
  }


  inline const CtgT* baseFac(size_t obsIdx) const {
    return &fac[(obsIdx - baseObs) * nPredFac];
  }


  inline const double* baseNum(size_t obsIdx) const {
    return &num[(obsIdx - baseObs) * nPredNum];
  }
};


#endif
