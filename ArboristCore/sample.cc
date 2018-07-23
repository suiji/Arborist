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
#include "splitpred.h"
#include "samplepred.h"

// Simulation-invariant values.
//
unsigned int Sample::nSamp = 0;

/**
 @brief Lights off initializations needed for sampling.

 @param nSamp_ is the number of samples.

 @return void.
*/
void Sample::Immutables(unsigned int nSamp_) {
  nSamp = nSamp_;
}


/**
   @brief Finalizer.

   @return void.
*/
void Sample::DeImmutables() {
  nSamp = 0;
}


Sample::Sample() :
  ctgRoot(vector<SumCount>(SampleNux::getNCtg())) {
}

    
Sample::~Sample() {
}


/**
   @brief Samples and counts occurrences of each target 'row'
   of the sampling vector.

   @param sCountRow outputs a vector of sample counts, by row.

   @return count of unique rows sampled:  bag count.
*/
unsigned int Sample::rowSample(vector<unsigned int> &sCountRow) {
  auto rvRow = CallBack::sampleRows(nSamp);
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
SampleCtg *Sample::FactoryCtg(const double y[], const RowRank *rowRank,  const unsigned int yCtg[], BV *treeBag) {
  SampleCtg *sampleCtg = new SampleCtg();
  sampleCtg->PreStage(yCtg, y, rowRank, treeBag);

  return sampleCtg;
}


/**
   @brief Static entry for regression response.

 */
SampleReg *Sample::FactoryReg(const double y[], const RowRank *rowRank, const unsigned int *row2Rank, BV *treeBag) {
  SampleReg *sampleReg = new SampleReg();
  sampleReg->PreStage(y, row2Rank, rowRank, treeBag);

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

   @param bagSum is the sum of in-bag sample values.  Used for initializing index tree root.

   @return void.
*/
void SampleReg::PreStage(const double y[], const unsigned int *row2Rank, const RowRank *rowRank, BV *treeBag) {
  vector<unsigned int> ctgProxy(rowRank->NRow());
  fill(ctgProxy.begin(), ctgProxy.end(), 0);
  Sample::PreStage(y, &ctgProxy[0], rowRank, treeBag);
  SetRank(row2Rank);
}


unique_ptr<SplitPred> SampleReg::SplitPredFactory(const FrameTrain *frameTrain, const RowRank *rowRank) const {
  return rowRank->SPRegFactory(frameTrain, bagCount);
}


/**
   @brief Compresses row->rank map to sIdx->rank.  Requires
   that row2Sample[] is complete:  PreStage().

   @param row2Rank[] is the response ranking, by row.

   @return void, with side-effected sample2Rank[].
 */
void SampleReg::SetRank(const unsigned int *row2Rank) {
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
SampleCtg::SampleCtg() : Sample() {
  SumCount scZero;
  scZero.Init();

  fill(ctgRoot.begin(), ctgRoot.end(), scZero);
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
void SampleCtg::PreStage(const unsigned int yCtg[], const double y[], const RowRank *rowRank, BV *treeBag) {
  Sample::PreStage(y, yCtg, rowRank, treeBag);
}


unique_ptr<SplitPred> SampleCtg::SplitPredFactory(const FrameTrain *frameTrain, const RowRank *rowRank) const {
  return rowRank->SPCtgFactory(frameTrain, bagCount, SampleNux::getNCtg());
}


/**
   @brief Sets the stage, so to speak, for a newly-sampled response set.

   @param y is the proxy / response:  classification / summary.

   @param yCtg is true response / zero:  classification / regression.

   @return void.
 */
void Sample::PreStage(const double y[], const unsigned int yCtg[], const RowRank *rowRank, BV *treeBag) {
  unsigned int nRow = rowRank->NRow();
  row2Sample = move(vector<unsigned int>(nRow));
  fill(row2Sample.begin(), row2Sample.end(), nSamp);

  vector<unsigned int> sCountRow(nRow);
  fill(sCountRow.begin(), sCountRow.end(), 0);
  bagCount = rowSample(sCountRow);
  sampleNode = move(vector<SampleNux>(bagCount));

  unsigned int slotBits = BV::SlotElts();
  bagSum = 0.0;
  unsigned int slot = 0;
  unsigned int sIdx = 0;
  for (unsigned int base = 0; base < nRow; base += slotBits, slot++) {
    unsigned int bits = 0ul;
    unsigned int mask = 1ul;
    unsigned int supRow = nRow < base + slotBits ? nRow : base + slotBits;
    for (unsigned int row = base; row < supRow; row++, mask <<= 1) {
      if (sCountRow[row] > 0) {
        row2Sample[row] = sIdx;
        bagSum += setNode(sIdx++, y[row], sCountRow[row], yCtg[row]);
        bits |= mask;
      }
    }
    treeBag->setSlot(slot, bits);
  }
}


/**
   @brief Invokes RowRank staging methods and caches compression map.

   @return void.
*/
unique_ptr<SamplePred> Sample::Stage(const RowRank *rowRank,
                          vector<StageCount> &stageCount) {
  auto samplePred = rowRank->SamplePredFactory(bagCount);
  //  samplePred->Stage(rowRank, sampleNode, row2Sample, stageCount);
  rowRank->Stage(sampleNode, row2Sample, samplePred.get(), stageCount);
  row2Sample.clear(); // No longer needed.
  
  return samplePred;
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
