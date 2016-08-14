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
#include "forest.h"

//#include <iostream>
using namespace std;

// Simulation-invariant values.
//
unsigned int Sample::nRow = 0;
unsigned int Sample::nPred = 0;
int Sample::nSamp = -1;

unsigned int SampleCtg::ctgWidth = 0;

/**
 @brief Lights off initializations needed for sampling.

 @param _nRow is the number of response/observation rows.

 @param _nSamp is the number of samples.

 @return void.
*/
void Sample::Immutables(unsigned int _nRow, unsigned int _nPred, int _nSamp, const double _feSampleWeight[], bool _withRepl, unsigned int _ctgWidth, int _nTree) {
  nRow = _nRow;
  nPred = _nPred;
  nSamp = _nSamp;
  CallBack::SampleInit(nRow, _feSampleWeight, _withRepl);
  if (_ctgWidth > 0)
    SampleCtg::Immutables(_ctgWidth, _nTree);
}


/**
   @return void.
 */
void SampleCtg::Immutables(unsigned int _ctgWidth, int _nTree) {
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
  nPred = 0;
  nSamp = -1;
  SampleCtg::DeImmutables();
}


Sample::Sample() {
  treeBag = new BV(nRow);
  row2Sample = new int[nRow];
  sampleNode = new SampleNode[nSamp]; // Lives until scoring.
}


Sample::~Sample() {
  delete treeBag;
  delete [] sampleNode;
  delete [] row2Sample;
  delete samplePred;
  delete bottom;
}


/**
   @brief Samples and enumerates instances of each row index.
   'bagCount'.

   @return vector of sample counts, by row.
*/
unsigned int *Sample::RowSample() {
  unsigned int *sCountRow = new unsigned int[nRow];
  for (unsigned int row = 0; row < nRow; row++) {
    sCountRow[row] = 0;
  }

  // Counts occurrences of the rank associated with each target 'row' of the
  // sampling vector.
  //
  int *rvRow = new int[nSamp];
  CallBack::SampleRows(nSamp, rvRow);
  for (int i = 0; i < nSamp; i++) {
    int row = rvRow[i];
    sCountRow[row]++;
  }
  delete [] rvRow;

  return sCountRow;
}


/**
   @brief Static entry for classification.
 */
SampleCtg *Sample::FactoryCtg(const std::vector<double> &y, const RowRank *rowRank,  const std::vector<unsigned int> &yCtg) {
  SampleCtg *sampleCtg = new SampleCtg();
  sampleCtg->Stage(yCtg, y, rowRank);

  return sampleCtg;
}


/**
   @brief Static entry for regression response.

 */
SampleReg *Sample::FactoryReg(const std::vector<double> &y, const RowRank *rowRank, const std::vector<unsigned int> &row2Rank) {
  SampleReg *sampleReg = new SampleReg();
  sampleReg->Stage(y, row2Rank, rowRank);

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
void SampleReg::Stage(const std::vector<double> &y, const std::vector<unsigned int> &row2Rank, const RowRank *rowRank) {
  std::vector<unsigned int> ctgProxy(nRow);
  std::fill(ctgProxy.begin(), ctgProxy.end(), 0);
  Sample::PreStage(y, ctgProxy, rowRank);
  SetRank(row2Rank);
  bottom = Bottom::FactoryReg(samplePred, bagCount);
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
    int sIdx = SampleIdx(row);
    if (sIdx >= 0)
      sample2Rank[sIdx] = row2Rank[row];
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
void SampleCtg::Stage(const std::vector<unsigned int> &yCtg, const std::vector<double> &y, const RowRank *rowRank) {
  Sample::PreStage(y, yCtg, rowRank);
  bottom = Bottom::FactoryCtg(samplePred, sampleNode, bagCount);
}


/**
   @brief Sets the stage, so to speak, for a newly-sampled response set.

   @param y is the proxy / response:  classification / summary.

   @param yCtg is true response / zero:  classification / regression.

   @return vector of compressed indices into sample data structures.
 */
void Sample::PreStage(const std::vector<double> &y, const std::vector<unsigned int> &yCtg, const RowRank *rowRank) {
  unsigned int *sCountRow = RowSample();
  unsigned int slotBits = BV::SlotElts();

  bagSum = 0.0;
  int slot = 0;
  unsigned int sIdx = 0;
  for (unsigned int base = 0; base < nRow; base += slotBits, slot++) {
    unsigned int bits = 0;
    unsigned int mask = 1;
    unsigned int supRow = nRow < base + slotBits ? nRow : base + slotBits;
    for (unsigned int row = base; row < supRow; row++, mask <<= 1) {
      unsigned int sCount = sCountRow[row];
      if (sCount > 0) {
        double val = sCount * y[row];
	sampleNode[sIdx].Set(val, sCount, yCtg[row]);
	bagSum += val;
        bits |= mask;
	row2Sample[row] = sIdx++;
      }
      else {
	row2Sample[row] = -1;
      }
    }
    treeBag->SetSlot(slot, bits);
  }
  bagCount = sIdx;
  delete [] sCountRow;

  samplePred = SamplePred::Factory(nPred, bagCount);
  PreStage(rowRank);
}


/**
   @brief Loops through the predictors to stage.

   @return void.
 */
void Sample::PreStage(const RowRank *rowRank) {
  int predIdx;

#pragma omp parallel default(shared) private(predIdx)
  {
#pragma omp for schedule(dynamic, 1)
    for (predIdx = 0; predIdx < int(nPred); predIdx++) {
      PreStage(rowRank, predIdx);
    }
  }
}


/**
   @brief Stages SamplePred objects in non-decreasing predictor order.

   @param predIdx is the predictor index.

   @return void.
*/
void Sample::PreStage(const RowRank *rowRank, int predIdx) {
  // TODO:  For sparse predictors, stage to DenseRank.

  // Predictor orderings recorded by RowRank may be built with an unstable sort.
  // Lookup() therefore need not map to 'idx', and results vary by predictor.
  //
  unsigned int spIdx = 0;
  std::vector<StagePack> stagePack(bagCount);
  for (unsigned int idx = 0; idx < nRow; idx++) {
    unsigned int predRank;
    unsigned int row = rowRank->Lookup(predIdx, idx, predRank);
    int sIdx = SampleIdx(row);
    if (sIdx >= 0) {
      unsigned int sCount;
      FltVal ySum;
      unsigned int ctg = Ref(sIdx, ySum, sCount);
      stagePack[spIdx++].Set(sIdx, predRank, sCount, ctg, ySum);
    }
  }
  samplePred->Stage(stagePack, predIdx);
}


void Sample::RowInvert(std::vector<unsigned int> &sample2Row) const {
  for (unsigned int row = 0; row < nRow; row++) {
    int sIdx = row2Sample[row];
    if (sIdx >= 0) {
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
