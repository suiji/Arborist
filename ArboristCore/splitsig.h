/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#ifndef ARBORIST_SPLITSIG_H
#define ARBORIST_SPLITSIG_H

#include "predictor.h"

// Transfers splitting information between levels as well as to decision trees.
// Reused from level to level, so there are only 2^levels allocated.
//
class SplitSig {
  int nodeMax; // Set from factory.
  static SplitSig *WSLookup(int predIdx, int accumOff);
 public:
  int sCount; // # samples subsumed by the post-split LHS.
  int lhEdge; // Far right index of LHS within buffer.
  double Gini; // Gini gain at split.
  static void Factory(const int _accumCount);
  static void ReFactory(const int _accumCount);
  static void DeFactory();
  static int ArgMaxGini(int liveCount, int nodeIdx, double preBias, double parGini, int &lhEdge, int &sCount);
  static double LHRH(const int pred, const int liveId, const int sourceOff, const int lhId, const int rhId);
  SplitSig *LevelSS(int predIdx, int liveCount, int nodeIdx) {
    int facIdx = Predictor::FacIdx(predIdx);
    return facIdx >= 0 ? levelSSFac + facIdx * liveCount + nodeIdx : levelSSNum + predIdx * liveCount + nodeIdx;
  }

  virtual double Replay(const int pred, const int liveId, const int sourceOff, const int lhId, const int rhId) = 0;
  static class SplitNode* Lower(const int, const double preBias, const int accumIdx, class SplitNode *par, const bool isLH);
  virtual class SplitNode* Lower(int predIdx, int accumIdx, double preBias, class SplitNode *par, const bool isLH) = 0;
};

class SplitSigNum : public SplitSig {
  SplitSigNum *levelSSNum;
  double Replay(const int pred, const int liveId, const int sourceOff, const int lhId, const int rhId);
 public:
  int spLow; // Lower predictor ordinal of split.
  int spHigh; // Higher predictor ordinal of split.
  class SplitNode *Lower(int _pred, int accumIdx, double preBias, class SplitNode *par, const bool isLH);
};

// Count and ordinal offset only needed until lowering, at which time the bits
// for the LHS are set in a temporary vector for the level at hand.
//
class SplitSigFac : public SplitSig {
  SplitSigFac *levelSSFac;
  double Replay(const int pred, const int liveId, const int sourceOff, const int lhId, const int rhId);
  static bool *treeSplitBits; // Accumulated factor-valued splits.
  static int treeBitOffset;  // Current offset into above.
  static bool* levelWSFacBits; // Level workspace for pre-reconciled LHS ordinals.
  static bool* LevelBits(int facIdx, int nodeIdx, int &facWith);
 public:
  int subset; // LHS bits.
  class SplitNode *Lower(int _pred, int nodeIdx, double preBias, class SplitNode *par, const bool isLH);
  static void Factory(int _nodeMax);
  static void ReFactory();
  static void DeFactory();
  static void TreeInit();
  static void FacBits(int facIdx, int nodeIdx, int lhsWidth, const int *facOrd);
  static void SingleBit(int facIdx, int nodeIdx, int ordIdx);
  static void Clear(int facIdx, int nodeIdx);
  static int SplitFacWidth();
  static void ConsumeTreeSplitBits(int *outBits);
};

// Looks up position and span of bits associated with factor / accumulator pair.
//
inline bool *SplitSigFac::LevelBits(int facIdx, int nodeIdx, int &facWidth) {
  facWidth = Predictor::FacWidth(facIdx);
  return levelWSFacBits + Predictor::FacOffset(facIdx)* nodeMax + nodeIdx * facWidth;
}

//#include <iostream>
//using namespace std;

// Sets the LHS bits in the level-wide workspace vector, in case the predictor is chosen
// for splitting at reconciliation.
//
inline void SplitSigFac::FacBits(int facIdx, int nodeIdx, int lhsWidth, const int *facOrd) {
  int facWidth;
  bool *lhsBits = LevelBits(facIdx, nodeIdx, facWidth);
  for (int i = 0; i < facWidth; i++)
    lhsBits[i] = false;
  for (int fc = 0; fc < lhsWidth; fc++) {
    lhsBits[facOrd[fc]] = true;
    //    cout << "Factor " << facIdx << "[ " << accumIdx << "] setting bit " << facOrd[fc] << endl;
  }
}

inline void SplitSigFac::Clear(int facIdx, int nodeIdx) {
  int facWidth;
  bool *lhsBits = LevelBits(facIdx, nodeIdx, facWidth);
  for (int i = 0; i < facWidth; i++)
    lhsBits[i] = false;
}


// Sets individual LHS bit in the level-wide workspace vector.
//
inline void SplitSigFac::SingleBit(int facIdx, int nodeIdx, int ordIdx) {
  int facWidth;
  bool *lhsBits = LevelBits(facIdx, nodeIdx, facWidth);
  lhsBits[ordIdx] = true;
}

#endif
