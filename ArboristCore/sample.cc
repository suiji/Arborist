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

void Sample::immutables(unsigned int nSamp_) {
  nSamp = nSamp_;
}


void Sample::deImmutables() {
  nSamp = 0;
}


Sample::Sample(const RowRank* rowRank_) :
  rowRank(rowRank_),
  ctgRoot(vector<SumCount>(SampleNux::getNCtg())),
  row2Sample(vector<unsigned int>(rowRank->getNRow())),
  bagCount(0),
  bagSum(0.0) {
}

    
Sample::~Sample() {
}


unsigned int Sample::rowSample(vector<unsigned int> &sCountRow) {
  vector<unsigned int> rvRow(CallBack::sampleRows(nSamp));

  fill(sCountRow.begin(), sCountRow.end(), 0);
  return countSamples(rvRow, sCountRow);
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

shared_ptr<SampleCtg> Sample::factoryCtg(const double y[],
                                         const RowRank *rowRank,
                                         const unsigned int yCtg[],
                                         BV *treeBag) {
  shared_ptr<SampleCtg> sampleCtg = make_shared<SampleCtg>(rowRank);
  sampleCtg->bagSamples(yCtg, y, treeBag);

  return sampleCtg;
}


shared_ptr<SampleReg> Sample::factoryReg(const double y[],
                                         const RowRank *rowRank,
                                         BV *treeBag) {
  shared_ptr<SampleReg> sampleReg = make_shared<SampleReg>(rowRank);
  sampleReg->bagSamples(y, treeBag);

  return sampleReg;
}


SampleReg::SampleReg(const RowRank *rowRank) : Sample(rowRank) {
}



void SampleReg::bagSamples(const double y[], BV *treeBag) {
  vector<unsigned int> ctgProxy(row2Sample.size());
  fill(ctgProxy.begin(), ctgProxy.end(), 0);
  Sample::bagSamples(y, &ctgProxy[0], treeBag);
}


unique_ptr<SplitNode> SampleReg::splitNodeFactory(const FrameTrain *frameTrain) const {
  return rowRank->SPRegFactory(frameTrain, bagCount);
}


SampleCtg::SampleCtg(const RowRank* rowRank) : Sample(rowRank) {
  SumCount scZero;
  scZero.init();

  fill(ctgRoot.begin(), ctgRoot.end(), scZero);
}


// Same as for regression case, but allocates and sets 'ctg' value, as well.
// Full row count is used to avoid the need to rewalk.
//
void SampleCtg::bagSamples(const unsigned int yCtg[], const double y[], BV *treeBag) {
  Sample::bagSamples(y, yCtg, treeBag);
}


unique_ptr<SplitNode> SampleCtg::splitNodeFactory(const FrameTrain *frameTrain) const {
  return rowRank->SPCtgFactory(frameTrain, bagCount, SampleNux::getNCtg());
}


void Sample::bagSamples(const double y[], const unsigned int yCtg[], BV *treeBag) {
  // Samples row indices and counts occurrences.
  //
  const unsigned int nRow = row2Sample.size();
  vector<unsigned int> sCountRow(nRow);
  bagCount = rowSample(sCountRow);

  // Copies contents of sampled outcomes and builds mapping vectors.
  //
  fill(row2Sample.begin(), row2Sample.end(), bagCount);
  const unsigned int slotBits = BV::SlotElts();
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


unique_ptr<SamplePred> Sample::predictors() const {
  return rowRank->SamplePredFactory(bagCount);
}


vector<StageCount> Sample::stage(SamplePred* samplePred) const {
  return move(samplePred->stage(rowRank, sampleNode, this));
}


SampleCtg::~SampleCtg() {
}


SampleReg::~SampleReg() {
}
