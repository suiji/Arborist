/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#ifndef ARBORIST_PRETREE_H
#define ARBORIST_PRETREE_H

class PTNode {
 public:
  int depth; // Level at which this node appears.
  int treeOff; // Offset of node from base of decision tree.
  int id; // DIAGNOSTIC
  class SplitNode* par; // All nodes but root have non-zero parent.
  PTNode(class SplitNode *par, const bool isLH);
};

typedef union { double num; int fac; } SplitVal;

// TODO:  deprecate 'tag' field.  For now, though, navigates offset in DecTree.
//
class SplitNode : public PTNode {
 public:
  int pred; // Predictor encoded in DecTree format.
  SplitVal sval; // Index or splitting val.
  double Gini;
  PTNode *rh; // Always nonzero.
  PTNode *lh; // Always nonzero.
  Bump bump; // Bump table entry.
 SplitNode(int _pred, SplitVal _sv, double _Gini, SplitNode *par, const bool isLH) : PTNode(par, isLH), lh(0), rh(0), pred(_pred), sval(_sv), Gini(_Gini) {}
};

class Leaf : public PTNode {
 public:
 Leaf(SplitNode *par, bool isLH) : PTNode(par, isLH) {}
  double score; // Regression only:  sum/sCount of accumulator predecessor.
};

// "Compressed" SplitNode, flushed out by Decompress().
//
class DevSplit {
 public:
  int pred;
  int parId;
  double Gini;
  int rhOff;
  int lhOff;
  bool isLH;
  char subset; // Taken directly from SSF.
};


class PreTree {
 public:
  static int rowBlock;
  static DevSplit *devSplit;
  static SplitNode **splitSet;
  static Leaf **leafSet;
  static SplitNode **parent; // Per-accumulator current parent.
  static SplitNode **parentNext;  // Parents for next level, by accumulator.
  static int *leafMap; // Sized for full (integer) leaf range.
  static int leafCount;
  static int splitCount;
  static void Factory(int nSamp, const int _accumCount, const int rowBlock);
  static void DeFactory();
  static void TreeInit(const int _bagCount, const int _acccumCount);
  static void DispatchQuantiles(const int treeSize, int leafPos[], int leafExtent[], int rank[], int rankCount[]);
  static int bagCount;
  static int *qOff;
  static int *qRanks;
  static double ParGini(const int liveIdx);
  static int Produce(const int levels);
  static SplitNode* AddSplit(const int liveIdx, int predIdx, char subset, double gini,  bool isLH);
  static SplitNode* AddSplit(SplitNode *par, int predIdx, char subset, double gini, bool isLH);
  static void SetParent(const int lhId, const int rhId, SplitNode *splitNode);
  static int AddLeaf(SplitNode *par, bool _isLH);
  static void AddLeaf(SplitNode *par, int leafId, bool _isLH);
  static int AddLeaf(const int liveIdx, class NodeCache *tfAccum);
  static void FlushLevel(const int countNext);
  static void ConsumeLeaves(double scoreVec[], int predVec[]);
  static void ConsumeSplits(double splitVec[], int predVec[], Bump bumpVec[]);
  static int TreeOffsets(const int levels);
};

#endif
