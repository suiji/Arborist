/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include "splitsig.h"
#include "train.h"
#include "predictor.h"
#include "level.h"
#include "node.h"
#include "facrun.h"
#include "pretree.h"

#include <iostream>
using namespace std;

SplitSigNum *SplitSigNum::levelSSNum = 0;

SplitSigFac *SplitSigFac::levelSSFac = 0;
int SplitSigFac::treeBitOffset = -1;
bool* SplitSigFac::levelWSFacBits = 0;
bool* SplitSigFac::treeSplitBits = 0;

//int *SplitSig::levelSampIdx = 0;

// N.B.:  '_accumCount' is only needed by SplitSigFac.
//
void SplitSig::Factory(int stCount) {
  nPredNum = Predictor::nPredNum();
  nPredFac = Predictor::nPredFac();
  if (nPredNum > 0)
    SplitSigNum::Factory(stCount);
  if (nPredFac > 0)
    SplitSigFac::Factory(stCount);
}

void SplitSig::ReFactory(int _nodeMax) {
  nodeMax = _nodeMax;
  if (nPredNum > 0)
    SplitSigNum::Refactory();
  if (nPredFac > 0)
    SplitSigFac::ReFactory();
}

void SplitSigFac::Factory() {
  treeBitOffset = 0;

  // Should be wide enough to hold all factor bits for a single level workspace:
  levelWSFacBits = new bool[nodeMax * Predictor::facTot];

  // Should be wide enough to hold all factor bits for an entire tree:
  treeSplitBits = new bool[2 * nodeMax * Predictor::facTot];

  levelSSFac = new SplitSigFac[nodeMax * nPredFac];
}

void SplitSigNum::Factory() {
  levelSSNum = new SplitSigNum[nodeMax * nPredNum];
}

void SplitSigNum::ReFactory() {
  delete [] levelSSNum; // Copy first?
  levelSSNum = new SplitSigNum[nodeMax * nPredNum];
}

SplitSigNum *SplitSigNum::SplitBase(int predIdx) {
  return levelSSNum + predIdx * stCount;
}

SplitSigFac *SplitSigFac::SplitBase(int facIdx) {
  return levelSSFac + facIdx * stCount;
}

void SplitSigFac::ReFactory() {
  delete [] levelWSFacBits;
  delete [] levelSSFac; // Copy first?

  // Nothing to be copied into the new workspace, as bits for this level
  // are already consumed.
  levelWSFacBits = new bool[nodeMax * Predictor::facTot];
  levelSSFac = new SplitSigFac[nodeMax * nPredFac];

  // Tree split bits accumulate, so data must be copied on realloc.
  bool *temp = new bool[2 * nodeMax * Predictor::facTot];
  for (int i = 0; i < treeBitOffset; i++)
    temp[i] = treeSplitBits[i];
  //  ConsumeTreeSplitBits(temp);  Cannot use:  expects integer target.

  delete [] treeSplitBits;
  treeSplitBits = temp;
}

void SplitSig::DeFactory() {
  if (nPredNum > 0) {
    SplitSigNum::DeFactory();
  }
  if (nPredFac > 0) {
    SplitSigFac::DeFactory();
  }
}

void SplitSigNum::DeFactory() {
  delete [] levelSSNum;
  leveSSNum = 0;
}

void SplitSigFac::DeFactory() {
  delete [] levelWSFacBits;
  delete [] treeSplitBits;
  delete [] levelSSFac;
  levelWSFacBits = 0;
  levelSSFac = 0;
  treeBitOffset = -1;
}

// Returns pointer to SplitSig in the level workspace.  Data only good during the
// lifetime of the level from which it is requested.
//
SplitSig *SplitSig::WSLookup(int predIdx, int nodeIdx) {
  int facIdx = Predictor::FacIdx(predIdx);
  if (facIdx >= 0) {
    return LevelElt::levelSSFac + facIdx * nodeMax + nodeIdx;
  }
  else {
    return LevelElt::levelSSNum + predIdx * nodeMax + nodeIdx;
  }
}

SplitNode *SplitSig::Lower(int predIdx, double preBias, int nodeIdx, SplitNode *par, bool isLH) {
  SplitSig *ss = WSLookup(predIdx, nodeIdx);
  return ss->Lower(predIdx, nodeIdx, preBias, par, isLH);
}

// Returns SplitSig with minimal information, in a form suitable for decision
// tree construction.
//
SplitNode *SplitSigNum::Lower(int pred, int nodeIdx, double preBias, SplitNode *par, const bool isLH) {
  double *predCol = Predictor::numBase + pred*Predictor::nRow;
  // N.B.:  Ordinals are rank equivalence classes and correspond to row number of
  // sorted predictor values.  The sorted values reside at 'numBase'.
  //
  //double low = x.column(pred)[DataOrd::rank2Row[spLow]];
  //double high = x.column(pred)[DataOrd::rank2Row[spHigh]];
  if (spLow < 0 || spLow > Predictor::nRow || spHigh <0 || spHigh > Predictor::nRow)
    cout << "NONSENSICAL split" << spLow << " / " << spHigh << " : " << Predictor::nRow << endl;
  double low = predCol[spLow];
  double high = predCol[spHigh];

  // DIAGNOSTICS:
  if (spLow == spHigh)
    cout << "TRIVIAL SPLIT " << spLow << " / " << spHigh << endl;
  else if (low > high)
    cout << "BAD SPLIT  (" << pred << ") "<<  low << " / " << high << " ords:  " << spLow << " / " << spHigh << "  " << Predictor::nRow << " rows" <<endl;
  else if (low == high)
    cout << "TIED SPLIT:  " << low << " / " << high << endl;

  SplitVal sval;
  sval.num = 0.5 * (low + high);
  //  cout << pred << "[" << accumIdx << "] " << sval.num << endl;

  return  new SplitNode(pred, sval, Gini-preBias, par, isLH);
}

// 
//
SplitNode *SplitSigFac::Lower(int predIdx, int nodeIdx, double preBias, SplitNode *par, const bool isLH) {
  SplitVal sval;
  int facIdx = Predictor::FacIdx(predIdx);

  // Copies LHS bits from static structure LEVEL-based vector.  These also go away with
  // the next level.
  int facWidth;
  bool *facBits = LevelBits(facIdx, nodeIdx, facWidth);
  bool *splitBits = treeSplitBits + treeBitOffset;
  for (int fc = 0; fc < facWidth; fc++) {
    splitBits[fc] = facBits[fc];
    //cout << "Predictor " << predIdx << ", " << fc << ":  " << facBits[fc] << " offset " << treeBitOffset << endl;
  }

  sval.num = treeBitOffset; // Records the TREE-based position of split information.
  treeBitOffset += facWidth;

  // Predictor offset should be -(1 + factor), to distinguish factors
  // from non-factors in the decision tree.
  //  
  return new SplitNode(-(1 + facIdx), sval, Gini-preBias, par, isLH);
}

double SplitSig::LHRH(const int predIdx, const int liveIdx, const int level, const int lhId, const int rhId) {
  SplitSig *ss = WSLookup(predIdx, liveIdx);
  return ss->Replay(predIdx, liveIdx, level, lhId, rhId);
}

double SplitSigNum::Replay(const int predIdx, const int liveIdx, const int level, const int lhId, const int rhId) {
  return NodeCache::SampleReplayLHRH(liveIdx, predIdx, level, lhEdge + 1);
}

// The LHS ordinals are recovered from lhsBit[], which is set by SplitSigFac::FacBit() whenever
// a splitting candidate appears.  These ordinals index the facRun[] structure, which in turn
// lists the runs among indices pertaining to each factor.  The LHS accumulator, 'lhId' can
// be set from these runs of indices.
double SplitSigFac::Replay(const int predIdx, const int liveIdx, const int level, const int lhId, const int rhId) {
  int facIdx = Predictor::FacIdx(predIdx);
  //cout << "Factor " << facIdx << " [" << lhAccum << "] " << endl;
  int facWidth;
  bool *lhsBits = LevelBits(facIdx, liveIdx, facWidth);
  //  cout << "Reading " << facIdx << " [" << lhAccum << "] up to " << facWidth << endl;
  double lhSum = 0.0;
  for (int fc = 0; fc < facWidth; fc++) {
    FacRun *fRun = FacRun::Ref(facIdx, liveIdx, fc);
    if (fRun->sCount == 0)
      continue;
    int start = fRun->start;
    int count = fRun->end - fRun->start + 1;
    if (lhsBits[fc]) {
      //cout << "ordinal " << fc << ":  " << start << " plus " << count << endl;
      lhSum += Node::SampleReplay(predIdx, level, fRun->start, count, lhId);
    }
    else {
      (void) Node::SampleReplay(predIdx, level, fRun->start, count, rhId); 
    }
  }
  return lhSum;
}


// Once per tree, resets the width offset count.
//
void SplitSigFac::TreeInit() {
  treeBitOffset = 0;
}

// After all SplitSigs in the tree are lowered, returns the total width of factors
// seen as splitting values.
//
int SplitSigFac::SplitFacWidth() {
  return treeBitOffset;
}


// Writes factor bits from all levels into contiguous vector.
// Deletes vectors of factor bits at each level.
//
void SplitSigFac::ConsumeTreeSplitBits(int *outBits) {
  for (int i = 0; i < treeBitOffset; i++) {
    outBits[i] = treeSplitBits[i]; // Upconvert to integer type for output to front end.
    //    cout << outBits[i] << endl;
  }
}

// Returns absolute predictor index of SplitSig with highest Gini gain greater than 'preBias',
// if any.
//
int SplitSig::ArgMaxGini(int liveCount, int nodeIdx, double preBias, double parGini, int &lhIdxCount, int &sCount) {
  int curPredIdx = -1;
  double curGini = preBias;

  // TODO:  Randomize predictor walk to break ties nondeterministically.  Body already
  // split to accomodate this:
  //
  for (int predIdx = 0; predIdx < Predictor::NPred(); predIdx++) {
    SplitSig *ss = LevelSS(predIdx, liveCount, nodeIdx);
    if (ss->lhEdge >= 0 && ss->Gini > curGini) {
      curGini = ss->Gini;
      curPredIdx = predIdx;
      sCount = ss->sCount;
      lhIdxCount = ss->lhEdge + 1;
    }
  }

  if (curPredIdx >= 0 && ((curGini - preBias) < Train::minRatio * parGini))
    curPredIdx = -1;

  return curPredIdx;
}

// Resets the SplitSigs for all live predictor nodes in the upcoming level.
//
void SplitSig::LevelReset(int liveCount) {
  SplitSigNum ssn;
  ssn.lhEdge = -1;
  ssn.sCount = 0;

  SplitSigFac ssf;
  ssf.lhEdge = -1;
  ssf.sCount = 0;

  // Can stride by 'liveCount' i.o. 'stCount', but using the larger
  // stride reduces chance of false sharing.
  //
  for (int nodeIdx = 0; nodeIdx < liveCount; nodeIdx++) {
    double preBias = Node::GetPrebias(nodeIdx);
    ssn.Gini = preBias;
    for (int i = 0; i < nPredNum; i++) {
      *(levelSSNum + i * stCount + nodeIdx) = ssn;
    }
    ssf.Gini = preBias;
    for (int i = 0; i < nPredFac; i++) {
      *(levelSSFac + i * stCount + nodeIdx)  = ssf;
    }
  }
}
