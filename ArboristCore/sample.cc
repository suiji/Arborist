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
void Sample::Immutables(unsigned int _nRow, unsigned int _nPred, int _nSamp, double _feSampleWeight[], bool _withRepl, unsigned int _ctgWidth, int _nTree) {
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
  sampleNode = new SampleNode[nSamp]; // Lives until scoring.
}


Sample::~Sample() {
  delete treeBag;
  delete [] sampleNode;
  delete samplePred;
  delete splitPred;
}


/**
   @brief Samples and enumerates instances of each row index.
   'bagCount'.

   @param sCountRow[] holds the row counts:  0 <=> OOB.

   @param sIdxRow[] row index into sample vector:  -1 <=> OOB.

   @return bagCount, plus output vectors.
*/
unsigned int Sample::CountRows(int sCountRow[], int sIdxRow[]) {
  for (unsigned int row = 0; row < nRow; row++) {
    sCountRow[row] = 0;
    sIdxRow[row] = -1;
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

  unsigned int idx = 0;
  for (unsigned int row = 0; row < nRow; row++) {
    if (sCountRow[row] > 0)
      sIdxRow[row] = idx++;
  }

  return idx;
}


/**
   @brief Static entry for classification.
 */
SampleCtg *SampleCtg::Factory(const std::vector<double> &y, const RowRank *rowRank,  const std::vector<unsigned int> &yCtg) {
  SampleCtg *sampleCtg = new SampleCtg();
  sampleCtg->Stage(yCtg, y, rowRank);

  return sampleCtg;
}


/**
   @brief Static entry for regression response.

 */
SampleReg *SampleReg::Factory(const std::vector<double> &y, const RowRank *rowRank, const std::vector<unsigned int> &row2Rank) {
  SampleReg *sampleReg = new SampleReg();
  sampleReg->Stage(y, row2Rank, rowRank);

  return sampleReg;
}


/**
   @brief Constructor.
 */
SampleReg::SampleReg() : Sample() {
  sample2Rank = new unsigned int[nSamp]; // Lives until scoring.
}



/**
   @brief Inverts the randomly-sampled vector of rows.

   @param y is the response vector.

   @param row2Rank is rank of each sampled row.

   @param samplePred is the SamplePred object associated with this tree.

   @param bagSum is the sum of in-bag sample values.  Used for initializing index tree root.

   @return count of in-bag samples.
*/
void SampleReg::Stage(const std::vector<double> &y, const std::vector<unsigned int> &row2Rank, const RowRank *rowRank) {
  std::vector<unsigned int> ctgProxy(nRow);
  std::fill(ctgProxy.begin(), ctgProxy.end(), 0);
  int *sIdxRow = Sample::PreStage(y, ctgProxy);

  // Only client is quantile regression.
  for (unsigned int row = 0; row < nRow; row++) {
    int sIdx = sIdxRow[row];
    if (sIdx >= 0) {
      sample2Rank[sIdx] = row2Rank[row];
    }
  }
  samplePred = SamplePred::Factory(rowRank, sampleNode, sIdxRow, nRow, nPred, bagCount);
  delete [] sIdxRow;

  splitPred = SplitPred::FactoryReg(samplePred);
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
  int *sIdxRow = Sample::PreStage(y, yCtg);
  samplePred = SamplePred::Factory(rowRank, sampleNode, sIdxRow, nRow, nPred, bagCount);
  delete [] sIdxRow;

  splitPred = SplitPred::FactoryCtg(samplePred, sampleNode);
}


/**
   @brief Sets the stage, so to speak, for a newly-sampled response set.

   @param y is the proxy / response:  classification / summary.

   @param yCtg is true response / zero:  classification / regression.

   @return vector of compressed indices into sample data structures.
 */
int *Sample::PreStage(const std::vector<double> &y, const std::vector<unsigned int> &yCtg) {
  int *sIdxRow = new int[nRow];
  int *sCountRow = new int[nRow];
  bagCount = CountRows(sCountRow, sIdxRow);
  unsigned int slotBits = BV::SlotBits();

  bagSum = 0.0;
  int slot = 0;
  for (unsigned int base = 0; base < nRow; base += slotBits, slot++) {
    unsigned int bits = 0;
    unsigned int mask = 1;
    unsigned int supRow = nRow < base + slotBits ? nRow : base + slotBits;
    for (unsigned int row = base; row < supRow; row++, mask <<= 1) {
      int sIdx = sIdxRow[row];
      if (sIdx >= 0) {
	int sCount = sCountRow[row];
        double val = sCount * y[row];
	sampleNode[sIdx].Set(val, sCount, yCtg[row]);
	bagSum += val;
        bits |= mask;
      }
    }
    treeBag->SetSlot(slot, bits);
  }
  delete [] sCountRow;

  return sIdxRow;
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
