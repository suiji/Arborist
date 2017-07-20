// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file sample.cc

   @brief Methods for sampling from the response to begin training an individual tree.

   @author Mark Seligman
 */

#include "sample.h"
#include "bv.h"
#include "callback.h"
#include "rowrank.h"
#include "samplepred.h"
#include "splitpred.h"
#include "bottom.h"
#include "index.h"

//#include <iostream>
//using namespace std;

// Simulation-invariant values.
//
unsigned int Sample::nSamp = 0;


/**
 @brief Lights off initializations needed for sampling.

 @param _nSamp is the number of samples.

 @return void.
*/
void Sample::Immutables(unsigned int _nSamp, const std::vector<double> &_feSampleWeight, bool _withRepl) {
  nSamp = _nSamp;
  CallBack::SampleInit(_feSampleWeight.size(), &_feSampleWeight[0], _withRepl);
}


/**
   @brief Finalizer.

   @return void.
*/
void Sample::DeImmutables() {
  nSamp = 0;
}


Sample::Sample(unsigned int _nRow, unsigned int nCtg) : treeBag(new BV(_nRow)), row2Sample(std::vector<unsigned int>(_nRow)), nRow(_nRow), noSample(nRow), ctgRoot(std::vector<SumCount>(nCtg)) {
  std::fill(row2Sample.begin(), row2Sample.end(), noSample);
  sampleNode.reserve(nSamp);
}


Sample::~Sample() {
  delete treeBag;
}


/**
   @brief Samples and counts occurrences of each target 'row'
   of the sampling vector.

   @param sCountRow outputs a vector of sample counts, by row.

   @return void.
*/
void Sample::RowSample(std::vector<unsigned int> &sCountRow) {
  std::vector<int> rvRow(nSamp);
  CallBack::SampleRows(nSamp, &rvRow[0]);
  for (auto row : rvRow) {
    sCountRow[row]++;
  }
}


/**
   @brief Static entry for classification.
 */
SampleCtg *Sample::FactoryCtg(const std::vector<double> &y, const RowRank *rowRank,  const std::vector<unsigned int> &yCtg, unsigned int _nCtg) {
  SampleCtg *sampleCtg = new SampleCtg(y.size(), _nCtg);
  sampleCtg->PreStage(yCtg, y, rowRank);

  return sampleCtg;
}


/**
   @brief Static entry for regression response.

 */
SampleReg *Sample::FactoryReg(const std::vector<double> &y, const RowRank *rowRank, const std::vector<unsigned int> &row2Rank) {
  SampleReg *sampleReg = new SampleReg(y.size());
  sampleReg->PreStage(y, row2Rank, rowRank);

  return sampleReg;
}


/**
   @brief Constructor.
 */
SampleReg::SampleReg(unsigned int _nRow) : Sample(_nRow, 0) {
}



/**
   @brief Inverts the randomly-sampled vector of rows.

   @param y is the response vector.

   @param row2Rank is the response ranking, by row.

   @param bagSum is the sum of in-bag sample values.  Used for initializing index tree root.

   @return void.
*/
void SampleReg::PreStage(const std::vector<double> &y, const std::vector<unsigned int> &row2Rank, const RowRank *rowRank) {
  std::vector<unsigned int> ctgProxy(nRow);
  std::fill(ctgProxy.begin(), ctgProxy.end(), 0);
  bagCount = Sample::PreStage(y, ctgProxy, rowRank);
  SetRank(row2Rank);
}


SplitPred *SampleReg::SplitPredFactory(const PMTrain *pmTrain, const RowRank *rowRank, const class Coproc *coproc) const {
  return SPRegFactory(coproc, pmTrain, rowRank, bagCount);
}


/**
   @brief Compresses row->rank map to sIdx->rank.  Requires
   that row2Sample[] is complete:  PreStage().

   @param row2Rank[] is the response ranking, by row.

   @return void, with side-effected sample2Rank[].
 */
void SampleReg::SetRank(const std::vector<unsigned int> &row2Rank) {
  // Only client is quantile regression.
  sample2Rank = new unsigned int[bagCount];
  for (unsigned int row = 0; row < nRow; row++) {
    unsigned int sIdx;
    if (SampleIdx(row, sIdx)) {
      sample2Rank[sIdx] = row2Rank[row];
    }
  }
}


/**
   @brief Constructor.
 */
SampleCtg::SampleCtg(unsigned int _nRow, unsigned int _nCtg) : Sample(_nRow, _nCtg), nCtg(_nCtg) {
  SumCount scZero;
  scZero.Init();

  std::fill(ctgRoot.begin(), ctgRoot.end(), scZero);
}


/**
   @brief Samples the response, sets in-bag bits and stages.

   @param yCtg is the response vector.

   @param y is the proxy response vector.

   @return void, with output vector parameter.
*/
// Same as for regression case, but allocates and sets 'ctg' value, as well.
// Full row count is used to avoid the need to rewalk.
//
void SampleCtg::PreStage(const std::vector<unsigned int> &yCtg, const std::vector<double> &y, const RowRank *rowRank) {
  bagCount = Sample::PreStage(y, yCtg, rowRank);
}


SplitPred *SampleCtg::SplitPredFactory(const PMTrain *pmTrain, const RowRank *rowRank, const class Coproc *coproc) const {
  return SPCtgFactory(coproc, pmTrain, rowRank, bagCount, nCtg);
}


/**
   @brief Sets the stage, so to speak, for a newly-sampled response set.

   @param y is the proxy / response:  classification / summary.

   @param yCtg is true response / zero:  classification / regression.

   @return count of SampleNodes built:  in-bag count.
 */
unsigned int Sample::PreStage(const std::vector<double> &y, const std::vector<unsigned int> &yCtg, const RowRank *rowRank) {
  std::vector<unsigned int> sCountRow(nRow);
  std::fill(sCountRow.begin(), sCountRow.end(), 0);
  RowSample(sCountRow);
  unsigned int slotBits = BV::SlotElts();

  bagSum = 0.0;
  int slot = 0;
  for (unsigned int base = 0; base < nRow; base += slotBits, slot++) {
    unsigned int bits = 0;
    unsigned int mask = 1;
    unsigned int supRow = nRow < base + slotBits ? nRow : base + slotBits;
    for (unsigned int row = base; row < supRow; row++, mask <<= 1) {
      if (sCountRow[row] > 0) {
	row2Sample[row] = sampleNode.size();
	bagSum += SetNode(y[row], sCountRow[row], yCtg[row]);
        bits |= mask;
      }
    }
    treeBag->SetSlot(slot, bits);
  }

  return sampleNode.size();
}


IndexLevel *Sample::IndexFactory(const PMTrain *pmTrain, const RowRank *rowRank, const Coproc *coproc) const {
  SamplePred *samplePred = SamplePredFactory(coproc, rowRank->NPred(), bagCount, rowRank->SafeSize(bagCount));
  SplitPred *splitPred = SplitPredFactory(pmTrain, rowRank, coproc);

  return new IndexLevel(samplePred, ctgRoot, new Bottom(pmTrain, rowRank, splitPred, samplePred, bagCount), nSamp, bagCount, bagSum);
}


/**
   @brief Loops through the predictors to stage.

   @return void.
 */
void Sample::Stage(const RowRank *rowRank, SamplePred *samplePred, Bottom *bottom) const {

  int predIdx;
#pragma omp parallel default(shared) private(predIdx)
  {
#pragma omp for schedule(dynamic, 1)
    for (predIdx = 0; predIdx < int(rowRank->NPred()); predIdx++) {
      Stage(rowRank, samplePred, bottom, predIdx);
    }
  }
}


/**
   @brief Stages SamplePred objects in non-decreasing predictor order.

   @param predIdx is the predictor index.

   @return void.
*/
void Sample::Stage(const RowRank *rowRank, SamplePred *samplePred, Bottom *bottom, unsigned int predIdx) const {
  std::vector<StagePack> stagePack;
  stagePack.reserve(bagCount); // Too big iff implicits present.
  unsigned int idxCount = rowRank->ExplicitCount(predIdx);
  for (unsigned int idx = 0; idx < idxCount; idx++) {
    unsigned int row, rank;
    rowRank->Ref(predIdx, idx, row, rank);
    PackIndex(row, rank, stagePack);
  }

  unsigned int extent;
  unsigned int safeOffset = rowRank->SafeOffset(predIdx, bagCount, extent);
  bool singleton = samplePred->Stage(stagePack, predIdx, safeOffset, extent);
  bottom->RootDef(predIdx, stagePack.size(), singleton);
}


/**
   @brief Packs rank with response statistics iff row is sampled.

   @return void.
 */
void Sample::PackIndex(unsigned int row, unsigned int predRank, std::vector<StagePack> &stagePack) const {
  unsigned int sIdx;
  if (SampleIdx(row, sIdx)) {
    StagePack packItem;
    unsigned int sCount;
    FltVal ySum;
    unsigned int ctg = Ref(sIdx, ySum, sCount);
    packItem.Init(sIdx, predRank, sCount, ctg, ySum);
    stagePack.push_back(packItem);
  }
}


void Sample::RowInvert(std::vector<unsigned int> &sample2Row) const {
  for (unsigned int row = 0; row < nRow; row++) {
    unsigned int sIdx;
    if (SampleIdx(row, sIdx)) {
      sample2Row[sIdx] = row;
    }
  }
}


SampleCtg::~SampleCtg() {
}


/**
   @brief Clears per-tree information.

   @return void.
 */
SampleReg::~SampleReg() {
  delete [] sample2Rank;
}
