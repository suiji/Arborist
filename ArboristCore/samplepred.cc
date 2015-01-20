// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include "samplepred.h"
#include "index.h"
#include "pretree.h"
#include "splitpred.h"

SamplePred *SamplePred::samplePredWS = 0;
int SamplePred::nSamp = 0;
int SamplePred::ctgShift = -1;

RestageMap *RestageMap::restageMap = 0;
int RestageMap::splitCount = -1;
int RestageMap::totLhIdx = -1;

void RestageMap::Factory(int levelMax) {
  restageMap = new RestageMap[levelMax];
  splitCount = 0;
}

// The map should not be in use when reallocating.
//
void RestageMap::ReFactory(int levelMax) {
  delete [] restageMap;
  restageMap = new RestageMap[levelMax];
}

void RestageMap::DeFactory() {
  delete [] restageMap;
  splitCount = -1;
}
 
void SamplePred::TreeInit(int nPred, int _nSamp) {
  nSamp = _nSamp;
  samplePredWS = new SamplePred[2 * nPred * nSamp];
}

void SamplePred::TreeClear() {
  delete [] samplePredWS;
  samplePredWS = 0;
  nSamp = -1;
}

// Returns the consecutive predictor ranks from which a split value can
// be computed.
//
void SamplePred::SplitRanks(int predIdx, int level, int spPos, int &rkLow, int &rkHigh) {
  SamplePred *tOrd = BufferOff(predIdx, level);
  rkLow = tOrd[spPos].rank;
  rkHigh = tOrd[spPos + 1].rank;
}

// Replays a block of SamplePreds for the predictor, remapping the PreTree index
// to 'ptId' for each sample index in the block.
//
// Returns sum of samples values associated with the block.
//
double SamplePred::Replay(int predIdx, int level, int start, int end, int ptId) {
  SamplePred *tOrd = BufferOff(predIdx, level);
  double sum = 0.0;

  for (int idx = start; idx <= end; idx++) {
    PreTree::MapSample(tOrd[idx].sampleIdx, ptId);
    sum += tOrd[idx].yVal;
  }

  return sum;
}

// Consume all information associated with current NodeCache item relevant to restaging.
//
void RestageMap::ConsumeSplit(int _splitIdx, int _lNext, int _rNext, int _lhIdxCount, int _rhIdxCount) {
  RestageMap *rsMap = &restageMap[_splitIdx];
  rsMap->lNext = _lNext;
  rsMap->rNext = _rNext;
  rsMap->lhIdxCount = _lhIdxCount;
  rsMap->rhIdxCount = _rhIdxCount;
  NodeCache::RestageFields(_splitIdx, rsMap->ptL, rsMap->ptR, rsMap->startIdx, rsMap->endIdx);
}

// The node cache still retains data for the splits handled in the previous
// level.  If these splits are walked in order, then, the left and right level
// indices should preserve the correct ordering for the next level.
//
// Returns the starting positiong for the next level.
//
void RestageMap::Restage(int predIdx, int level) {
  int lhIdx = 0;
  int rhIdx = totLhIdx;
  SamplePred *source = SamplePred::BufferOff(predIdx, level-1);
  SamplePred *targ = SamplePred::BufferOff(predIdx, level);

  for (int splitIdx = 0; splitIdx < splitCount; splitIdx++) {
    RestageMap *rsMap = &restageMap[splitIdx];
    if (SplitPred::PredRun(splitIdx, predIdx, level-1)) {
      rsMap->TransmitRun(splitIdx, predIdx, level-1);
    }
    else {
      rsMap->Restage(source, targ, lhIdx, rhIdx);
      rsMap->NoteRuns(predIdx, level, targ, lhIdx, rhIdx);
    }
    lhIdx += rsMap->lhIdxCount;
    rhIdx += rsMap->rhIdxCount;
  }
  // Diagnostic:
  // rhIdx == rhStop + 1
}

// Runs of minimal length should not make it to restaging, so no effort is made to threshold
// by run length.
//
void RestageMap::NoteRuns(int predIdx, int level, const SamplePred targ[], int lhIdx, int rhIdx) {
  if (lNext >= 0) {
    if (targ[lhIdx].rank == targ[lhIdx + lhIdxCount - 1].rank)
      SplitPred::SetPredRun(lNext, predIdx, level, true);
  }

  if (rNext >= 0) {
    if (targ[rhIdx].rank == targ[rhIdx + rhIdxCount - 1].rank)
      SplitPred::SetPredRun(rNext, predIdx, level, true);
  }
}

// Runs are maintained by SplitPred, as SamplePred does not record either
// of split or predictor.
//
// Runs need not be restaged, but lh, rh index positions should be
// updated for uniformity across predictors.  Hence the data in
// unrestaged SamplePreds is dirty.
//
void RestageMap::TransmitRun(int splitIdx, int predIdx, int level) {
  SplitPred::TransmitRun(splitIdx, predIdx, lNext, rNext, level);
}

// Restaging is implemented as a stable partition, first grouping the left-hand
// subnodes, then the right.
//
void RestageMap::Restage(const SamplePred source[], SamplePred targ[], int lhIdx, int rhIdx) {
  // Node either does not split or splits into two terminals.
  if (lNext < 0 && rNext < 0)
    return;

  if (lNext >= 0 && rNext >= 0) // Both subnodes potentially splitable.
    SamplePred::RestageTwo(source, targ, startIdx, endIdx, ptL, ptR, lhIdx, rhIdx);
  else if (lNext >= 0) // Only LH subnode potentially splitable.
    SamplePred::RestageOne(source, targ, startIdx, endIdx, ptL, lhIdx);
  else // Only RH subnode potentially splitable.
    SamplePred::RestageOne(source, targ, startIdx, endIdx, ptR, rhIdx);

  // Post-conditions:  lhIdx = lhIdx in + lhIdxCount && rhIdxIdx = rhIdx in + rhIdxCount
}

// Target nodes should all equal either lh or rh.
//
void SamplePred::RestageTwo(const SamplePred source[], SamplePred targ[], int startIdx, int endIdx, int ptL, int ptR, int lhIdx, int rhIdx) {
  for (int i = startIdx; i <= endIdx; i++) {
    int sIdx = source[i].sampleIdx;
    int ptIdx = PreTree::Sample2PT(sIdx);
    if (ptIdx == ptL)
      targ[lhIdx++] = source[i];
    else {
      // if (ptIdx != ptR) // ASSERTION
      //	cout << "Expected " << ptR << ":  " << ptIdx << " sIndex " << sIdx << endl;;
      targ[rhIdx++] = source[i];
    }
  }
}

void SamplePred::RestageOne(const SamplePred source[], SamplePred targ[], int startIdx, int endIdx, int pt, int idx) {
  // Target nodes should all be either lh or leaf.
  for (int i = startIdx; i <= endIdx; i++) {
    int sIdx = source[i].sampleIdx;
    if (PreTree::Sample2PT(sIdx) == pt)
      targ[idx++] = source[i];
  }
}
