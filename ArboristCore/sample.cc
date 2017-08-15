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
#include "response.h"
#include "splitpred.h"

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


Sample::Sample(unsigned int _nRow, unsigned int nCtg) : treeBag(new BV(_nRow)), ctgRoot(std::vector<SumCount>(nCtg)) {
}


Sample::~Sample() {
  delete treeBag;
}


/**
   @brief Samples and counts occurrences of each target 'row'
   of the sampling vector.

   @param sCountRow outputs a vector of sample counts, by row.

   @return count of unique rows sampled:  bag count.
*/
unsigned int Sample::RowSample(std::vector<unsigned int> &sCountRow) {
  std::vector<int> rvRow(nSamp);
  CallBack::SampleRows(nSamp, &rvRow[0]);
  unsigned int _bagCount = 0;
  for (auto row : rvRow) {
    unsigned int sCount = sCountRow[row];
    _bagCount += sCount == 0 ? 1 : 0;
    sCountRow[row] = sCount + 1;
  }

  return _bagCount;
}


/**
   @brief Static entry for classification.
 */
SampleCtg *Sample::FactoryCtg(const std::vector<double> &y, const RowRank *rowRank,  const std::vector<unsigned int> &yCtg, unsigned int _nCtg, std::vector<unsigned int> &row2Sample) {
  SampleCtg *sampleCtg = new SampleCtg(y.size(), _nCtg);
  sampleCtg->PreStage(yCtg, y, rowRank, row2Sample);

  return sampleCtg;
}


/**
   @brief Static entry for regression response.

 */
SampleReg *Sample::FactoryReg(const std::vector<double> &y, const RowRank *rowRank, const std::vector<unsigned int> &row2Rank, std::vector<unsigned int> &row2Sample) {
  SampleReg *sampleReg = new SampleReg(y.size());
  sampleReg->PreStage(y, row2Rank, rowRank, row2Sample);

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
void SampleReg::PreStage(const std::vector<double> &y, const std::vector<unsigned int> &row2Rank, const RowRank *rowRank, std::vector<unsigned int> &row2Sample) {
  std::vector<unsigned int> ctgProxy(rowRank->NRow());
  std::fill(ctgProxy.begin(), ctgProxy.end(), 0);
  Sample::PreStage(y, ctgProxy, rowRank, row2Sample);
  SetRank(row2Sample, row2Rank);
}


SplitPred *SampleReg::SplitPredFactory(const PMTrain *pmTrain, const RowRank *rowRank) const {
  return rowRank->SPRegFactory(pmTrain, bagCount);
}


/**
   @brief Compresses row->rank map to sIdx->rank.  Requires
   that row2Sample[] is complete:  PreStage().

   @param row2Rank[] is the response ranking, by row.

   @return void, with side-effected sample2Rank[].
 */
void SampleReg::SetRank(const std::vector<unsigned int> &row2Sample, const std::vector<unsigned int> &row2Rank) {
  // Only client is quantile regression.
  sample2Rank = new unsigned int[bagCount];
  for (unsigned int row = 0; row < row2Sample.size(); row++) {
    unsigned int sIdx = row2Sample[row];
    if (sIdx < bagCount) {
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
void SampleCtg::PreStage(const std::vector<unsigned int> &yCtg, const std::vector<double> &y, const RowRank *rowRank, std::vector<unsigned int> &row2Sample) {
  Sample::PreStage(y, yCtg, rowRank, row2Sample);
}


SplitPred *SampleCtg::SplitPredFactory(const PMTrain *pmTrain, const RowRank *rowRank) const {
  return rowRank->SPCtgFactory(pmTrain, bagCount, nCtg);
}


/**
   @brief Sets the stage, so to speak, for a newly-sampled response set.

   @param y is the proxy / response:  classification / summary.

   @param yCtg is true response / zero:  classification / regression.

   @return void.
 */
void Sample::PreStage(const std::vector<double> &y, const std::vector<unsigned int> &yCtg, const RowRank *rowRank, std::vector<unsigned int> &row2Sample) {
  unsigned int nRow = rowRank->NRow();
  std::vector<unsigned int> sCountRow(nRow);
  std::fill(sCountRow.begin(), sCountRow.end(), 0);
  bagCount = RowSample(sCountRow);
  sampleNode = std::move(std::vector<SampleNode>(bagCount));

  unsigned int slotBits = BV::SlotElts();
  bagSum = 0.0;
  int slot = 0;
  unsigned int sIdx = 0;
  for (unsigned int base = 0; base < nRow; base += slotBits, slot++) {
    unsigned int bits = 0;
    unsigned int mask = 1;
    unsigned int supRow = nRow < base + slotBits ? nRow : base + slotBits;
    for (unsigned int row = base; row < supRow; row++, mask <<= 1) {
      if (sCountRow[row] > 0) {
	row2Sample[row] = sIdx;
	bagSum += SetNode(sIdx++, y[row], sCountRow[row], yCtg[row]);
        bits |= mask;
      }
    }
    treeBag->SetSlot(slot, bits);
  }
}


/**
   @brief Allocates primary objects employed in training a single tree.

   @return
 */
void Sample::StageFactory(const PMTrain *pmTrain, const RowRank *rowRank, const Response *response, Sample *&sample, SplitPred *&splitPred, SamplePred *&samplePred, std::vector<StageCount> &stageCount) {
  std::vector<unsigned int> row2Sample(rowRank->NRow());
  std::fill(row2Sample.begin(), row2Sample.end(), nSamp);
  sample = response->RootSample(rowRank, row2Sample);
  sample->Stage(rowRank, row2Sample, samplePred, stageCount);

  splitPred = sample->SplitPredFactory(pmTrain, rowRank);
}


/**
   @brief Invokes RowRank staging methods and caches compression map.

   @return void.
 */
void Sample::Stage(const RowRank *rowRank, const std::vector<unsigned int> &row2Sample, SamplePred *&samplePred, std::vector<StageCount> &stageCount) {
  samplePred = rowRank->SamplePredFactory(bagCount);
  rowRank->Stage(sampleNode, row2Sample, samplePred, stageCount);
  RowInvert(row2Sample);
}


/**
   @brief Builds the sample2Row[] map for unpacking by Leaf methods.

   @param row2Sample[] maps bagged rows to their respective sample indices.

   @return void.
*/
void Sample::RowInvert(const std::vector<unsigned int> &row2Sample) {
  sample2Row = std::move(std::vector<unsigned int>(bagCount));
  for (unsigned int row = 0; row < row2Sample.size(); row++) {
    unsigned int sIdx = row2Sample[row];
    if (sIdx < nSamp) {
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
