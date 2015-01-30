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

/**
   @brief Reallocates the restaging map.

   @param levelMax is the current level-max value.

   @return void.
 */
// The map should not be in use when reallocating.
//
void RestageMap::ReFactory(int levelMax) {
  delete [] restageMap;
  restageMap = new RestageMap[levelMax];
}

/**
   @brief Class finalizer.

   @return void.
 */
void RestageMap::DeFactory() {
  delete [] restageMap;
  splitCount = -1;
}
 
/**
   @brief Per-tree allocation of workspace.

   @param nPred is the number of predictors.

   @param _nSamp is the number of samples

   @return void.
 */
void SamplePred::TreeInit(int nPred, int _nSamp) {
  nSamp = _nSamp;
  samplePredWS = new SamplePred[2 * nPred * nSamp];
}

/**
   @brief Per-tree deallocations of workspace.

   @return void
 */
void SamplePred::TreeClear() {
  delete [] samplePredWS;
  samplePredWS = 0;
  nSamp = -1;
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
  SamplePred *tOrd = BufferOff(predIdx, level);
  rkLow = tOrd[spPos].rank;
  rkHigh = tOrd[spPos + 1].rank;
}

/**
   @brief Remaps the pretree index of a SamplePred block.

   @param predIdx is the splitting predictor.

   @param level is the current level.

   @param start is the block starting index.

   @param end is the block ending index.

   @param ptId is the pretree index to which to map the block.

   @return sum of response values associated with each replayed index.
*/
double SamplePred::Replay(int predIdx, int level, int start, int end, int ptId) {
  SamplePred *tOrd = BufferOff(predIdx, level);
  double sum = 0.0;

  for (int idx = start; idx <= end; idx++) {
    PreTree::MapSample(tOrd[idx].sampleIdx, ptId);
    sum += tOrd[idx].yVal;
  }

  return sum;
}

/**
   @brief Consumes all fields in current NodeCache item relevant to restaging.

   @param _splitIdx is the split index.

   @param _lNext is the index node offset of the LHS in the next level.

   @param _rNext is the index node offset of the RHS in the next level.

   @param _lhIdxCount is the count of indices associated with the split's LHS.

   @param _rhIdxCount is the count of indices associated with the split's RHS.

   @return void.
*/
void RestageMap::ConsumeSplit(int _splitIdx, int _lNext, int _rNext, int _lhIdxCount, int _rhIdxCount) {
  RestageMap *rsMap = &restageMap[_splitIdx];
  rsMap->lNext = _lNext;
  rsMap->rNext = _rNext;
  rsMap->lhIdxCount = _lhIdxCount;
  rsMap->rhIdxCount = _rhIdxCount;
  NodeCache::RestageFields(_splitIdx, rsMap->ptL, rsMap->ptR, rsMap->startIdx, rsMap->endIdx);
}

/**
   @brief Walks the live split indices for a predictor and either restages or propagates runs.

   @param predIdx is the predictor being restaged.

   @param level is the current level.

   @return void.
 */
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

/**
   @brief Notes those index nodes for consist of single runs.

   @param predIdx is the predictor index.
  
   @param level is the current level.

   @param targ contains the restaged indices for this predictor.

   @param lhIdx is the index of the LHS.

   @param rhIdx is the index of the RHS.

   @return void.
*/
// Runs of subminimal length should not make it to restaging, so no effort is
// made to threshold by run length.
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

/**
   @brief Transmits the bits associated with a run from the previous level into those for the descendents in the current level.

   @param splitIdx is the index of the node in the previous level.

   @param predIdx is the predictor for which the indices are restaged.

   @param level is the previous level.
   
   @return void.
 */
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

/**
   @brief Sends contents of previous level's SamplePreds to this level's descendents, via a stable partition.

   @param source contains the previous level's SamplePreds.

   @param targ outputs this level's SamplePreds.

   @param lhIdx is the index node offset for the LHS.

   @param rhIdx is the index node offset for the RHS.

   @return void, with output parameter vector.
*/
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

/**
   @brief Sends SamplePred contents to both LH and RH targets.

   @param source contains the previous level's SamplePred values.

   @param targ outputs the current level's SamplePred values.

   @param startIdx is the first index in the node being restaged.

   @param endIdx is the last index in the node being restaged.

   @param ptL is the pretree index of the LHS.

   @param ptR is the pretree index of the RHS.

   @param lhIdx is the index node offset of the LHS.

   @param rhIdx is the index node offset of the RHS.

   @return void.
 */
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

/**
   @brief Sends SamplePred contents to one of either LH or RH targets.

   @param source contains the previous level's SamplePred values.

   @param targ outputs the current level's SamplePred values.

   @param startIdx is the first index in the node being restaged.

   @param endIdx is the last index in the node being restaged.

   @param ptL is the pretree index of the descendent index node.

   @param idx is the offset of the descendent index node.

   @return void.
 */
void SamplePred::RestageOne(const SamplePred source[], SamplePred targ[], int startIdx, int endIdx, int pt, int idx) {
  // Target nodes should all be either lh or leaf.
  for (int i = startIdx; i <= endIdx; i++) {
    int sIdx = source[i].sampleIdx;
    if (PreTree::Sample2PT(sIdx) == pt)
      targ[idx++] = source[i];
  }
}
