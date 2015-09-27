// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file samplepred.cc

   @brief Methods to maintain predictor-wise orderings of sampled response indices.

   @author Mark Seligman
 */

#include "samplepred.h"
#include "sample.h"
#include "predictor.h"
#include "pretree.h"
#include "restage.h"
//#include <iostream>
using namespace std;

unsigned int SamplePred::nRow = 0;
unsigned int SamplePred::pitchSP = 0;
unsigned int SamplePred::pitchSIdx = -1;
int SamplePred::predNumFirst = -1;
int SamplePred::predNumSup = -1;
int SamplePred::predFacFirst = -1;
int SamplePred::predFacSup = -1;

int SamplePred::nSamp = -1;
int SamplePred::nPred = -1;
int SamplePred::bufferSize = -1;

unsigned int SPNode::runShift = 0;

/**
  @brief Sets static allocation parameters.

  @param _nPred is the number of predictors.

  @param _nSamp is the number of samples.

  @param _nRow is the number of rows.

  @param ctgWidth is the response cardinality.

  @return void.
 */
void SamplePred::Immutables(int _nPred, int _nSamp, unsigned int _nRow, unsigned int ctgWidth) {
  nPred = _nPred;
  nSamp = _nSamp;
  nRow = _nRow;

  predNumFirst = Predictor::NumFirst();
  predNumSup = Predictor::NumSup();
  predFacFirst = Predictor::FacFirst();
  predFacSup = Predictor::FacSup();
  
  // 'bagCount' suffices, but easier to preinitialize to same value for each tree:
  pitchSP = nSamp * sizeof(SamplePred);
  pitchSIdx = nSamp * sizeof(unsigned int);
  
  bufferSize = nPred * nSamp;
  SPNode::Immutables(ctgWidth);
  RestageMap::Immutables(nPred, nSamp);
}

/**
   @brief Computes a packing width sufficient to hold all (zero-based) response
   category values.

   @param ctgWidth is the response cardinality.

   @return void.
 */
void SPNode::Immutables(unsigned int ctgWidth) {
  unsigned int bits = 1;
  runShift = 0;
  // Ctg values are zero-based, so the first power of 2 greater than or
  // equal to 'ctgWidth' has sufficient bits to hold all response values.
  while (bits < ctgWidth) {
    bits <<= 1;
    runShift++;
  }
}


/*
**/
void SPNode::DeImmutables() {
  runShift = 0;
}


/**
   @brief Nothing to see here.
 */
void SamplePred::DeImmutables() {
  pitchSP = pitchSIdx = 0;
  nRow = 0;
  predNumFirst = predNumSup = predFacFirst = predFacSup = -1;
  nPred = nSamp = bufferSize = -1;
  SPNode::DeImmutables();
  RestageMap::DeImmutables();
}


/**
   @brief Base class constructor.
 */
SamplePred::SamplePred() {
  sampleIdx = new unsigned int[2* bufferSize];
  nodeVec = new SPNode[2 * bufferSize];
}


/**
  @brief Base class destructor.
 */
SamplePred::~SamplePred() {
  delete [] nodeVec;
  delete [] sampleIdx;
}

/**
   @brief Records ranked regression sample information per predictor.

   @return void.
 */
void SamplePred::StageReg(const PredOrd *predOrd, const SampleNode sampleReg[], const int sCountRow[], const int sIdxRow[]) {
  int predIdx;
#pragma omp parallel default(shared) private(predIdx)
    {
#pragma omp for schedule(dynamic, 1)
      for (predIdx = predNumFirst; predIdx < predNumSup; predIdx++) {
	StageReg(predOrd + predIdx * nRow, sampleReg, sCountRow, sIdxRow, predIdx);
      }
    }

#pragma omp parallel default(shared) private(predIdx)
    {
#pragma omp for schedule(dynamic, 1)
      for (predIdx = predFacFirst; predIdx < predFacSup; predIdx++) {
	StageReg(predOrd + predIdx * nRow, sampleReg, sCountRow, sIdxRow, predIdx);
      }
    }
}


/**
   @brief Stages the regression sample for a given predictor.  For each predictor derives rank associated with sampled row and random vector index.

   @param predIdx is the predictor index.

   @return void.
*/
void SamplePred::StageReg(const PredOrd dCol[], const SampleNode sampleReg[], const int sCountRow[], const int sIdxRow[], int predIdx) {
  unsigned int *sampleIdx;
  SPNode *spn = Buffers(predIdx, 0, sampleIdx);

  // 'rk' values must be recorded in nondecreasing rank order.
  //
  int ptIdx = 0;
  for (unsigned int rk = 0; rk < nRow; rk++) {
    PredOrd dc = dCol[rk];
    unsigned int row = dc.row;
    if (sCountRow[row] > 0) {
      int sIdx = sIdxRow[row];
      SampleNode sReg = sampleReg[sIdx];
      sampleIdx[ptIdx] = sIdx;
      spn[ptIdx++].SetReg(sReg.sum, dc.rank, sReg.sCount);
    }
  }
}


/**
   @brief Records ranked categorical sample information per predictor.

   @return void.
 */
void SamplePred::StageCtg(const PredOrd *predOrd, const SampleNodeCtg sampleCtg[], const int sCountRow[], const int sIdxRow[]) {
  
  int predIdx;
#pragma omp parallel default(shared) private(predIdx)
    {
#pragma omp for schedule(dynamic, 1)
      for (predIdx = predNumFirst; predIdx < predNumSup; predIdx++) {
	StageCtg(predOrd + predIdx * nRow, sampleCtg, sCountRow, sIdxRow, predIdx);
      }
    }

#pragma omp parallel default(shared) private(predIdx)
    {
#pragma omp for schedule(dynamic, 1)
      for (predIdx = predFacFirst; predIdx < predFacSup; predIdx++) {
	StageCtg(predOrd + predIdx * nRow, sampleCtg, sCountRow, sIdxRow, predIdx);
      }
    }
}


/**
   @brief Stages the categorical sample for a given predictor.

   @param predIdx is the predictor index.

   @return void.
*/
void SamplePred::StageCtg(const PredOrd dCol[], const SampleNodeCtg sampleCtg[], const int sCountRow[], const int sIdxRow[], int predIdx) {
  unsigned int *sampleIdx;
  SPNode *spn = Buffers(predIdx, 0, sampleIdx);

  // 'rk' values must be recorded in nondecreasing rank order.
  //
  int ptIdx = 0;
  for (unsigned int rk = 0; rk < nRow; rk++) {
    PredOrd dc = dCol[rk];
    unsigned int row = dc.row;
    if (sCountRow[row] > 0) {
      int sIdx = sIdxRow[row];
      SampleNodeCtg sCtg = sampleCtg[sIdx];
      sampleIdx[ptIdx] = sIdx;
      spn[ptIdx++].SetCtg(sCtg.sum, dc.rank, sCtg.sCount, sCtg.ctg);
    }
  }
}


/**
   @brief Fills in the high and low ranks defining a numerical split.

   @param predIdx is the splitting predictor.

   @param level is the current level.

   @param spPos is the index position of the split.

   @param rkLow outputs the low rank.

   @param rkHigh outputs the high rank.

   @return void, with output reference parameters.
 */
void SamplePred::SplitRanks(int predIdx, int level, int spPos, int &rkLow, int &rkHigh) {
  SPNode *spn = SplitBuffer(predIdx, level);
  rkLow = spn[spPos].Rank();
  rkHigh = spn[spPos + 1].Rank();
}


/**
   @brief Maps a block of predictor-associated sample indices to a specified pretree node index.

   @param predIdx is the splitting predictor.

   @param level is the current level.

   @param start is the block starting index.

   @param end is the block ending index.

   @param ptId is the pretree node index to which to map the block.

   @return sum of response values associated with each replayed index.
*/
double SamplePred::Replay(int sample2PT[], int predIdx, int level, int start, int end, int ptId) {
  unsigned int *sIdx;
  SPNode *spn = Buffers(predIdx, level, sIdx);

  double sum = 0.0;
  for (int idx = start; idx <= end; idx++) {
    sum += spn[idx].YVal();
    unsigned int sId = sIdx[idx];
    sample2PT[sId] = ptId;
  }

  return sum;
}
