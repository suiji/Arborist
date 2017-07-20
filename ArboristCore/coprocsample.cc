// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file coprocsample.cc

   @brief Bottom factory wrappers, parametrized by coprocessor state.

   @author Mark Seligman
 */

#include "coproc.h"

#include "sample.h"
#include "splitpred.h"
#include "samplepred.h"
#include "rowrank.h"
#include "predblock.h"


/**
   @brief Static entry for sample staging.

   @return SamplePred object for tree.
 */
SamplePred *Sample::SamplePredFactory(const Coproc *coproc, unsigned int _nPred, unsigned int _bagCount, unsigned int _bufferSize) {
  return new SamplePred(_nPred, _bagCount, _bufferSize);
}


SPCtg *Sample::SPCtgFactory(const Coproc *coproc, const PMTrain *pmTrain, const RowRank *rowRank, unsigned int bagCount, unsigned int _nCtg) {
  return new SPCtg(pmTrain, rowRank, bagCount, _nCtg);
}


SPReg *Sample::SPRegFactory(const Coproc *coproc, const PMTrain *pmTrain, const RowRank *rowRank, unsigned int bagCount) {
  return new SPReg(pmTrain, rowRank, bagCount);
}
