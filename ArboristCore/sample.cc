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
#include "bottom.h"

//#include <iostream>
//using namespace std;

// Simulation-invariant values.
//
unsigned int Sample::nRow = 0;
unsigned int Sample::nSamp = 0;

unsigned int SampleCtg::ctgWidth = 0;

/**
 @brief Lights off initializations needed for sampling.

 @param _nRow is the number of response/observation rows.

 @param _nSamp is the number of samples.

 @return void.
*/
void Sample::Immutables(unsigned int _nSamp, const std::vector<double> &_feSampleWeight, bool _withRepl, unsigned int _ctgWidth, unsigned int _nTree) {
  nRow = _feSampleWeight.size();
  nSamp = _nSamp;
  CallBack::SampleInit(nRow, &_feSampleWeight[0], _withRepl);
  if (_ctgWidth > 0)
    SampleCtg::Immutables(_ctgWidth, _nTree);
}


/**
   @return void.
 */
void SampleCtg::Immutables(unsigned int _ctgWidth, unsigned int _nTree) {
  ctgWidth = _ctgWidth;
}


/**
 */
void SampleCtg::DeImmutables() {
  ctgWidth = 0;
}


/**
   @brief Finalizer.

   @return void.
*/
void Sample::DeImmutables() {
  nRow = 0;
  nSamp = 0;
  SampleCtg::DeImmutables();
}


Sample::Sample() : treeBag(new BV(nRow)), row2Sample(std::vector<unsigned int>(nRow)), noSample(nRow) {
  std::fill(row2Sample.begin(), row2Sample.end(), noSample);
  sampleNode.reserve(nSamp);
}


Sample::~Sample() {
  delete treeBag;
  delete samplePred;
  delete bottom;
}


/**
   @brief Samples and counts occurrences of each target 'row'
   of the sampling vector.

   @param sCountRow outputs a vector of sample counts, by row.

   @return void.
*/
void Sample::RowSample(std::vector<unsigned int> &sCountRow) {
  int *rvRow = new int[nSamp];
  CallBack::SampleRows(nSamp, rvRow);
  for (unsigned int i = 0; i < nSamp; i++) {
    int row = rvRow[i];
    sCountRow[row]++;
  }

  delete [] rvRow;
}


/**
   @brief Static entry for classification.
 */
SampleCtg *Sample::FactoryCtg(const PMTrain *pmTrain, const std::vector<double> &y, const RowRank *rowRank,  const std::vector<unsigned int> &yCtg) {
  SampleCtg *sampleCtg = new SampleCtg();
  sampleCtg->Stage(pmTrain, yCtg, y, rowRank);

  return sampleCtg;
}


/**
   @brief Static entry for regression response.

 */
SampleReg *Sample::FactoryReg(const PMTrain *pmTrain, const std::vector<double> &y, const RowRank *rowRank, const std::vector<unsigned int> &row2Rank) {
  SampleReg *sampleReg = new SampleReg();
  sampleReg->Stage(pmTrain, y, row2Rank, rowRank);

  return sampleReg;
}


/**
   @brief Constructor.
 */
SampleReg::SampleReg() : Sample() {
}



/**
   @brief Inverts the randomly-sampled vector of rows.

   @param y is the response vector.

   @param row2Rank is the response ranking, by row.

   @param samplePred is the SamplePred object associated with this tree.

   @param bagSum is the sum of in-bag sample values.  Used for initializing index tree root.

   @return count of in-bag samples.
*/
void SampleReg::Stage(const PMTrain *pmTrain, const std::vector<double> &y, const std::vector<unsigned int> &row2Rank, const RowRank *rowRank) {
  std::vector<unsigned int> ctgProxy(nRow);
  std::fill(ctgProxy.begin(), ctgProxy.end(), 0);
  bagCount = Sample::PreStage(y, ctgProxy, rowRank, samplePred);
  bottom = Bottom::FactoryReg(pmTrain, rowRank, samplePred, bagCount);
  Sample::Stage(rowRank);
  SetRank(row2Rank);
}


/**
   @brief Compresses row->rank map to sIdx->rank.

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
SampleCtg::SampleCtg() : Sample() {
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
void SampleCtg::Stage(const PMTrain *pmTrain, const std::vector<unsigned int> &yCtg, const std::vector<double> &y, const RowRank *rowRank) {
  bagCount = Sample::PreStage(y, yCtg, rowRank, samplePred);
  bottom = Bottom::FactoryCtg(pmTrain, rowRank, samplePred, sampleNode, bagCount);
  Sample::Stage(rowRank);
}


/**
   @brief Sets the stage, so to speak, for a newly-sampled response set.

   @param y is the proxy / response:  classification / summary.

   @param yCtg is true response / zero:  classification / regression.

   @return bagCount value.
 */
unsigned int Sample::PreStage(const std::vector<double> &y, const std::vector<unsigned int> &yCtg, const RowRank *rowRank, SamplePred *&_samplePred) {
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
      unsigned int sCount = sCountRow[row];
      if (sCount > 0) {
        double val = sCount * y[row];
	row2Sample[row] = sampleNode.size();
	SampleNode sNode;
	sNode.Set(val, sCount, yCtg[row]);
	sampleNode.push_back(sNode);
	bagSum += val;
        bits |= mask;
      }
    }
    treeBag->SetSlot(slot, bits);
  }

  unsigned int sIdx = sampleNode.size();
  _samplePred = SamplePred::Factory(rowRank->NPred(), sIdx, rowRank->SafeSize(sIdx));
  return sIdx;
}


/**
   @brief Loops through the predictors to stage.

   @return void.
 */
void Sample::Stage(const RowRank *rowRank) {
  int predIdx;

#pragma omp parallel default(shared) private(predIdx)
  {
#pragma omp for schedule(dynamic, 1)
    for (predIdx = 0; predIdx < int(rowRank->NPred()); predIdx++) {
      Stage(rowRank, predIdx);
    }
  }
}


/**
   @brief Stages SamplePred objects in non-decreasing predictor order.

   @param predIdx is the predictor index.

   @return void.
*/
void Sample::Stage(const RowRank *rowRank, unsigned int predIdx) {
  std::vector<StagePack> stagePack;
  stagePack.reserve(bagCount); // Too big iff implicits present.
  unsigned int idxCount = rowRank->ExplicitCount(predIdx);
  for (unsigned int idx = 0; idx < idxCount; idx++) {
    unsigned int row, rank;
    rowRank->Ref(predIdx, idx, row, rank);
    PackIndex(row, rank, stagePack);
  }
  bottom->RootDef(predIdx, bagCount - stagePack.size());

  unsigned int extent;
  unsigned int safeOffset = rowRank->SafeOffset(predIdx, bagCount, extent);
  samplePred->Stage(stagePack, predIdx, safeOffset, extent);
}


/**
   @brief Packs rank with response statisitcs iff row is sampled.

   @return void.
 */
void Sample::PackIndex(unsigned int row, unsigned int predRank, std::vector<StagePack> &stagePack) {
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
