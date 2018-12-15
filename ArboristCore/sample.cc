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
#include "splitnode.h"
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
  ctgRoot(vector<SumCount>(SampleNux::getNCtg())),
  bagCount(0),
  bagSum(0.0) {
}

    
Sample::~Sample() {
}


/**
   @brief Samples and counts occurrences of each sampled row
   index.

   @param nRow is the number of rows in the training set.

   @param[out] bagCount_ is the count of bagged rows.

   @return row-indexed vector of sample counts.
*/
vector<unsigned int> Sample::rowSample(unsigned int nRow, unsigned int &bagCount_) {
  vector<unsigned int> rvRow(CallBack::sampleRows(nSamp));

  vector<unsigned int> sCountRow(nRow);
  fill(sCountRow.begin(), sCountRow.end(), 0);
  bagCount_ = countSamples(rvRow, sCountRow);

  return move(sCountRow);
}


vector<unsigned int> Sample::binIndices(const vector<unsigned int>& idx) {
  // Sets binPop to respective bin population, then accumulates population
  // of bins to the left.
  // Performance not sensitive to bin width.
  //
  vector<unsigned int> binPop(1 + binIdx(idx.size()));
  fill(binPop.begin(), binPop.end(), 0);
  for (auto val : idx) {
    binPop[binIdx(val)]++;
  }
  for (unsigned int i = 1; i < binPop.size(); i++) {
    binPop[i] += binPop[i-1];
  }

  // Available index initialzed to one less than total population left of and
  // including bin.  Empty bins have same initial index as bin to the left.
  // This is not a problem, as empty bins are not (re)visited.
  //
  vector<int> idxAvail(binPop.size());
  for (unsigned int i = 0; i < idxAvail.size(); i++) {
    idxAvail[i] = static_cast<int>(binPop[i]) - 1;
  }

  // Writes to the current available index for bin, which is then decremented.
  //
  // Performance degrades if bin width exceeds available cache.
  //
  vector<unsigned int> idxBinned(idx.size());
  for (auto index : idx) {
    int destIdx = idxAvail[binIdx(index)]--;
    idxBinned[destIdx] = index;
  }

  return move(idxBinned);
}


// Sample counting is sensitive to locality.  In the absence of
// binning, access is random.  Larger bins improve locality, but
// performance begins to degrade when bin size exceeds available
// cache.
unsigned int Sample::countSamples(vector<unsigned int>& idx,
                                  vector<unsigned int>& sc) {
  if (binIdx(sc.size()) > 0) {
    idx = move(binIndices(idx));
  }
    
  unsigned int nz = 0;
  for (auto index : idx) {
    nz += (sc[index] == 0 ? 1 : 0);
    sc[index]++;
  }

  return nz;
}

/**
   @brief Static entry for classification.
 */
SampleCtg *Sample::FactoryCtg(const double y[], const RowRank *rowRank,  const unsigned int yCtg[], BV *treeBag) {
  SampleCtg *sampleCtg = new SampleCtg();
  sampleCtg->preStage(yCtg, y, rowRank, treeBag);

  return sampleCtg;
}


/**
   @brief Static entry for regression response.

 */
SampleReg *Sample::FactoryReg(const double y[], const RowRank *rowRank, const unsigned int *row2Rank, BV *treeBag) {
  SampleReg *sampleReg = new SampleReg();
  sampleReg->preStage(y, row2Rank, rowRank, treeBag);

  return sampleReg;
}


/**
   @brief Constructor.
 */
SampleReg::SampleReg() : Sample() {
}



void SampleReg::preStage(const double y[], const unsigned int *row2Rank, const RowRank *rowRank, BV *treeBag) {
  vector<unsigned int> ctgProxy(rowRank->NRow());
  fill(ctgProxy.begin(), ctgProxy.end(), 0);
  Sample::preStage(y, &ctgProxy[0], rowRank, treeBag);
  setRank(row2Rank);
}


unique_ptr<SplitNode> SampleReg::SplitNodeFactory(const FrameTrain *frameTrain, const RowRank *rowRank) const {
  return rowRank->SPRegFactory(frameTrain, bagCount);
}


void SampleReg::setRank(const unsigned int *row2Rank) {
  // Only client is quantile regression.
  sample2Rank = new unsigned int[bagCount];
  for (unsigned int row = 0; row < row2Sample.size(); row++) {
    unsigned int sIdx;
    if (sampledRow(row, sIdx)) {
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


// Same as for regression case, but allocates and sets 'ctg' value, as well.
// Full row count is used to avoid the need to rewalk.
//
void SampleCtg::preStage(const unsigned int yCtg[], const double y[], const RowRank *rowRank, BV *treeBag) {
  Sample::preStage(y, yCtg, rowRank, treeBag);
}


unique_ptr<SplitNode> SampleCtg::SplitNodeFactory(const FrameTrain *frameTrain, const RowRank *rowRank) const {
  return rowRank->SPCtgFactory(frameTrain, bagCount, SampleNux::getNCtg());
}


/**
   @brief Sets the stage, so to speak, for a newly-sampled response set.

   @param y is the proxy / response:  classification / summary.

   @param yCtg is true response / zero:  classification / regression.

   @return void.
 */
void Sample::preStage(const double y[], const unsigned int yCtg[], const RowRank *rowRank, BV *treeBag) {
  unsigned int nRow = rowRank->NRow();
  vector<unsigned int> sCountRow(rowSample(nRow, bagCount));
  row2Sample = move(vector<unsigned int>(nRow));
  fill(row2Sample.begin(), row2Sample.end(), bagCount);

  unsigned int slotBits = BV::SlotElts();
  unsigned int slot = 0;
  unsigned int sIdx = 0;
  for (unsigned int base = 0; base < nRow; base += slotBits, slot++) {
    unsigned int bits = 0ul;
    unsigned int mask = 1ul;
    unsigned int supRow = nRow < base + slotBits ? nRow : base + slotBits;
    for (unsigned int row = base; row < supRow; row++, mask <<= 1) {
      if (sCountRow[row] > 0) {
        row2Sample[row] = sIdx++;
        bagSum += addNode(y[row], sCountRow[row], yCtg[row]);
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
unique_ptr<SamplePred> Sample::stage(const RowRank *rowRank,
                          vector<StageCount> &stageCount) {
  auto samplePred = rowRank->SamplePredFactory(bagCount);
  samplePred->stage(rowRank, sampleNode, this, stageCount);
  
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
