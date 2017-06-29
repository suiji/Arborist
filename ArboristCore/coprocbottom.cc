// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file coprocBottom.cc

   @brief Bottom factory wrappers, parametrized by coprocessor state.

   @author Mark Seligman
 */

#include "coproc.h"

#include "bottom.h"
#include "splitpred.h"
#include "samplepred.h"
#include "rowrank.h"
#include "predblock.h"


/**
   @brief Static entry for sample staging.

   @return SamplePred object for tree.
 */
SamplePred *Bottom::FactorySamplePred(const Coproc *coproc, unsigned int _nPred, unsigned int _bagCount, unsigned int _bufferSize) {
  return new SamplePred(_nPred, _bagCount, _bufferSize);
}


SPCtg *Bottom::FactorySPCtg(const Coproc *coproc, const PMTrain *pmTrain, const RowRank *rowRank, SamplePred *samplePred, const std::vector<SampleNode> &sampleCtg, unsigned int bagCount) {
  return new SPCtg(pmTrain, rowRank, samplePred, sampleCtg, bagCount);
}


SPReg *Bottom::FactorySPReg(const Coproc *coproc, const PMTrain *pmTrain, const RowRank *rowRank, SamplePred *samplePred, unsigned int bagCount) {
  return new SPReg(pmTrain, rowRank, samplePred, bagCount);
}
