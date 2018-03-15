// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file rowrank.cc

   @brief Methods for predictor-specific training.

   @author Mark Seligman
 */

#include "rowrank.h"
#include "framemap.h"
#include "sample.h"
#include "samplepred.h"
#include "splitpred.h"

#include <algorithm>

// Observations are blocked according to type.  Blocks written in separate
// calls from front-end interface.

/**
   @brief Constructor for row, rank passed from front end as parallel arrays.

   @param feRow is the vector of rows allocated by the front end.

   @param feRank is the vector of ranks allocated by the front end.

 */
RowRank::RowRank(const FrameTrain *frameTrain,
		 const unsigned int feRow[],
		 const unsigned int feRank[],
		 const unsigned int feRLE[],
		 unsigned int rleLength,
		 double _autoCompress) :
  nRow(frameTrain->NRow()),
  nPred(frameTrain->NPred()),
  noRank(max(nRow, frameTrain->CardMax())),
  nPredDense(0),
  denseIdx(vector<unsigned int>(nPred)),
  nonCompact(0),
  accumCompact(0),
  denseRank(vector<unsigned int>(nPred)),
  explicitCount(vector<unsigned int>(nPred)),
  rrStart(vector<unsigned int>(nPred)),
  safeOffset(vector<unsigned int>(nPred)),
  autoCompress(_autoCompress) {
  unsigned int explCount = DenseBlock(feRank, feRLE, rleLength);
  ModeOffsets();

  rrNode = move(vector<RRNode>(explCount));
  Decompress(feRow, feRank, feRLE, rleLength);
}


/**
   @brief Walks the design matrix as RLE entries, merging adjacent
   entries with identical ranks.

   @brief feRank are the ranks corresponding to runlength-encoding (RLE)
   entries.

   @param feRLE are the run lengths corresponding to RLE entries.

   @param rleLength is the count of RLE entries.

   @return total count of explicit slots.
 */
unsigned int RowRank::DenseBlock(const unsigned int feRank[], const unsigned int feRLE[], unsigned int rleLength) {
  unsigned int explCount = 0;
  unsigned int rleIdx = 0;
  for (unsigned int predIdx = 0; predIdx < nPred; predIdx++) {
    unsigned int denseMax = 0; // Running maximum of run counts.
    unsigned int argMax = noRank;
    unsigned int runCount = 0; // Runs across adjacent rle entries.
    unsigned int rankPrev = noRank;
    unsigned int rank;
    unsigned int runLength = RunSlot(feRLE, feRank, rleIdx, rank);

    for (unsigned int rowTot = runLength; rowTot <= nRow; rowTot += runLength) {
      if (rank == rankPrev) {
	runCount += runLength;
      }
      else {
	runCount = runLength;
	rankPrev = rank;
      }
      if (runCount > denseMax) {
	denseMax = runCount;
	argMax = rank;
      }
      if (++rleIdx == rleLength)
	break;
      runLength = RunSlot(feRLE, feRank, rleIdx, rank);
    }
    // Post condition:  rowTot == nRow.

    explCount += DenseMode(predIdx, denseMax, argMax);
  }

  return explCount;
}


/**
   @brief Determines whether predictor to be stored densely and updates
   storage accumulators accordingly.

   @param predIdx is the predictor under consideration.

   @param denseMax is the highest run length encountered for the predictor:
   must lie within [1, nRow].

   @param argMax is an argmax rank value corresponding to denseMax.

   @return void.
 */
unsigned int RowRank::DenseMode(unsigned int predIdx, unsigned int denseMax, unsigned int argMax) {
  unsigned int rowCount;
  if (denseMax > autoCompress * nRow) { // Sufficiently long run found.
    denseRank[predIdx] = argMax;
    safeOffset[predIdx] = accumCompact; // Accumulated offset:  dense.
    rowCount = nRow - denseMax;
    accumCompact += rowCount;
    denseIdx[predIdx] = nPredDense++;
  }
  else {
    denseRank[predIdx] = noRank;
    denseIdx[predIdx] = nPred; // Inattainable index. 
    safeOffset[predIdx] = nonCompact++; // Index:  non-dense storage.
    rowCount = nRow;
  }
  explicitCount[predIdx] = rowCount;

  return rowCount;
}


/**
   @brief Assigns predictor offsets according to storage mode:
   noncompressed predictors stored first, as with staging offsets.

   @return void.
 */
void RowRank::ModeOffsets() {
  unsigned int denseBase = nonCompact * nRow;
  for (unsigned int predIdx = 0; predIdx < nPred; predIdx++) {
    unsigned int offSafe = safeOffset[predIdx];
    rrStart[predIdx] = denseRank[predIdx] != noRank ? denseBase + offSafe :
      offSafe * nRow;
  }
}


/**
   @brief Decompresses a block of predictors deemed not to be storable
   densely.

   @param feRow[] are the rows corresponding to distinct runlength-
   encoded (RLE) entries.

   @param feRank[] are the ranks corresponing to RLE entries.

   @param feRLE records the run lengths spanning the original design
   matrix.

   @param rleLength is the total count of RLE entries.

   @return void.
 */
void RowRank::Decompress(const unsigned int feRow[], const unsigned int feRank[], const unsigned int feRLE[], unsigned int rleLength) {
  unsigned int rleIdx = 0;
  for (unsigned int predIdx = 0; predIdx < nPred; predIdx++) {
    unsigned int outIdx = rrStart[predIdx];
    unsigned int row, rank;
    unsigned int runLength = RunSlot(feRLE, feRow, feRank, rleIdx, row, rank);
    for (unsigned int rowTot = runLength; rowTot <= nRow; rowTot += runLength) {
      if (rank != denseRank[predIdx]) { // Non-dense runs expanded.
	for (unsigned int i = 0; i < runLength; i++) {
	  rrNode[outIdx++].Init(row + i, rank);
	}
      }
      if (++rleIdx == rleLength)
	break;
      runLength = RunSlot(feRLE, feRow, feRank, rleIdx, row, rank);
    }
    //    if (outIdx - rrStart[predIdx] != explicitCount[predIdx])
    //cout << "Dense count mismatch" << endl;
  }
}


/**
   @brief Destructor.
 */
RowRank::~RowRank() {
}


/**
   @brief Loops through the predictors to stage.

   @return void.
 */
void RowRank::Stage(const vector<SampleNux>  &sampleNode, const vector<unsigned int> &row2Sample, SamplePred *samplePred, vector<StageCount> &stageCount) const {
  int predIdx;
#pragma omp parallel default(shared) private(predIdx)
  {
#pragma omp for schedule(dynamic, 1)
    for (predIdx = 0; predIdx < int(nPred); predIdx++) {
      Stage(sampleNode, row2Sample, samplePred, predIdx, stageCount[predIdx]);
    }
  }
}

// RowRank must export vectors containing safe offset, extent, node-start and explicit count i/o to stage transparently:

/**
   @brief Stages SamplePred objects in non-decreasing predictor order.

   @param predIdx is the predictor index.

   @return void.
*/
void RowRank::Stage(const vector<SampleNux> &sampleNode,
		    const vector<unsigned int> &row2Sample,
		    SamplePred *samplePred,
		    unsigned int predIdx,
		    StageCount &stageCount) const {
  unsigned int extent;
  unsigned int safeOffset = SafeOffset(predIdx, samplePred->BagCount(), extent);

  stageCount.expl = samplePred->Stage(sampleNode, &rrNode[rrStart[predIdx]], row2Sample, ExplicitCount(predIdx), predIdx, safeOffset, extent, stageCount.singleton);
}


/**
   @brief Static entry for sample staging.

   @return SamplePred object for tree.
 */
unique_ptr<SamplePred> RowRank::SamplePredFactory(unsigned int _bagCount) const {
  return make_unique<SamplePred>(nPred, _bagCount, SafeSize(_bagCount));
}


unique_ptr<SPCtg> RowRank::SPCtgFactory(const FrameTrain *frameTrain,
			     unsigned int bagCount,
			     unsigned int _nCtg) const {
  return make_unique<SPCtg>(frameTrain, this, bagCount, _nCtg);
}


unique_ptr<SPReg> RowRank::SPRegFactory(const FrameTrain *frameTrain,
			     unsigned int bagCount) const {
  return make_unique<SPReg>(frameTrain, this, bagCount);
}


RankedPre::RankedPre(unsigned int _nRow,
		       unsigned int _nPredNum,
		       unsigned int _nPredFac) :
  nRow(_nRow),
  nPredNum(_nPredNum),
  nPredFac(_nPredFac),
  rank(vector<unsigned int>(0)),
  row(vector<unsigned int>(0)),
  runLength(vector<unsigned int>(0)),
  numOff(vector<unsigned int>(nPredNum)),
  numVal(vector<double>(0)) {
}


void RankedPre::NumSparse(const double feValNum[],
			   const unsigned int feRowStart[],
			   const unsigned int feRunLength[]) {
  unsigned int colOff = 0;
  for (unsigned int numIdx = 0; numIdx < nPredNum; numIdx++) {
    numOff[numIdx] = numVal.size();
    unsigned int idxCol = NumSortSparse(&feValNum[colOff], &feRowStart[colOff], &feRunLength[colOff]);
    colOff += idxCol;
  }
}

unsigned int RankedPre::NumSortSparse(const double feColNum[],
				       const unsigned int feRowStart[],
				       const unsigned int feRunLength[]) {
  vector<NumRLE> rleNum;
  for (unsigned int rleIdx = 0, rowTot = 0; rowTot < nRow; rowTot += feRunLength[rleIdx++]) {
    rleNum.push_back(make_tuple(feColNum[rleIdx], feRowStart[rleIdx], feRunLength[rleIdx]));
  }

  sort(rleNum.begin(), rleNum.end()); // runlengths silent, as rows unique.
  RankNum(rleNum);

  return rleNum.size();
}

void RankedPre::RankNum(const vector<NumRLE> &rleNum) {
  NumRLE elt = rleNum[0];
  unsigned int rk = 0;
  rank.push_back(rk);
  numVal.push_back(get<0>(elt));
  row.push_back(get<1>(elt));
  runLength.push_back(get<2>(elt));
  for (unsigned int idx = 1; idx < rleNum.size(); idx++) {
    elt = rleNum[idx];
    double valThis = get<0>(elt);
    unsigned int rowThis = get<1>(elt);
    unsigned int runCount = get<2>(elt);
    if (valThis == numVal.back() && rowThis == row.back() + runLength.back()) {
      runLength.back() += runCount;
    }
    else { // New RLE, rank entries regardless whether tied.
      if (valThis != numVal.back()) {
	rk++;
	numVal.push_back(valThis);
      }
      rank.push_back(rk);
      row.push_back(rowThis);
      runLength.push_back(runCount);
    }
  }
}


void RankedPre::NumDense(const double _feNum[]) {
  for (unsigned int numIdx = 0; numIdx < nPredNum; numIdx++) {
    numOff[numIdx] = numVal.size();
    NumSortRaw(&_feNum[numIdx * nRow]);
  }
}


void RankedPre::NumSortRaw(const double colNum[]) {
  vector<ValRowD> valRow(nRow);
  for (unsigned int row = 0; row < nRow; row++) {
    valRow[row] = make_pair(colNum[row], row);
  }

  sort(valRow.begin(), valRow.end());  // Stable sort.
  RankNum(valRow);
}

/**
   @brief Stores ordered predictor column, entering uncompressed.

   @param numOut outputs the rank-ordered predictor values.

   @return void.
 */
void RankedPre::RankNum(const vector<ValRowD> &valRow) {
  unsigned int rk = 0;
  runLength.push_back(1);
  row.push_back(valRow[0].second);
  numVal.push_back(valRow[0].first);
  rank.push_back(rk);
  for (unsigned int idx = 1; idx < valRow.size(); idx++) {
    double valThis = valRow[idx].first;
    unsigned int rowThis = valRow[idx].second;

    if (valThis == numVal.back() && rowThis == row.back() + runLength.back()) {
      runLength.back()++;
    }
    else { // New RLE, row and rank entries regardless whether tied.
      if (valThis != numVal.back()) {
	rk++;
	numVal.push_back(valThis);
      }
      rank.push_back(rk);
      runLength.push_back(1);
      row.push_back(rowThis);
    }
  }
}


void RankedPre::FacDense(const unsigned int feFac[]) {
  // Builds the ranked factor block.  Assumes 0-justification has been 
  // performed by bridge.
  //
  for (unsigned int facIdx = 0; facIdx < nPredFac; facIdx++) {
    FacSort(&feFac[facIdx * nRow]);
  }
}


/**
   @brief Sorts factors and stores as rank-ordered run-length encoding.

   @return void.
 */
void RankedPre::FacSort(const unsigned int predCol[]) {
  vector<ValRowI> valRow(nRow);
  for (unsigned int row = 0; row < nRow; row++) {
    valRow[row] = make_pair(predCol[row], row);
  }
  sort(valRow.begin(), valRow.end()); // Stable sort.
  RankFac(valRow);
}


/**
   @brief Builds rank-ordered run-length encoding to hold factor values.

   Final "rank" values are the internal factor codes and may contain
   gaps.  A dense numbering scheme would entail backmapping at LH bit
   assignment following splitting (q.v.):  prediction and training
   must map to the same factor levels.

   @return void.
*/ 
void RankedPre::RankFac(const vector<ValRowI> &valRow) {
  unsigned int rankPrev = valRow[0].first;
  unsigned int rowPrev = valRow[0].second;
  runLength.push_back(1);
  rank.push_back(rankPrev);
  row.push_back(rowPrev);
  for (unsigned int rowIdx = 1; rowIdx < valRow.size(); rowIdx++) {
    unsigned int rankThis = valRow[rowIdx].first;
    unsigned int rowThis = valRow[rowIdx].second;

    if (rankThis == rankPrev && rowThis == (rowPrev + 1)) {
      runLength.back() ++;
    }
    else {
      runLength.push_back(1);
      rank.push_back(rankThis);
      row.push_back(rowThis);
    }
    rankPrev = rankThis;
    rowPrev = rowThis;
  }
}

