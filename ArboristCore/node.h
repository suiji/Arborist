/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#ifndef ARBORIST_NODE_H
#define ARBORIST_NODE_H

// Node fields associated with the response, viz., invariant across
// predictors.
//
class Node {
  Node *Node;
  int *sample2Node;
  int probSize;
  int totNodes;
 protected:
  int nodeMax; // High-watermark for allocation and re-allocation.
 public: // The three integer values are all non-negative.
  int lhStart; // Start index of LHS data in buffer.
  int idxCount; // # distinct indices in the node.
  int sCount;  // # samples subsumed by this node.
  bool isLH; //  Root does not use.
  double sum; // Sum of all responses in node.
  double preBias; // Inf of Gini values eligible for splitting.
  class SplitNode *par;
  static void Factory(int _nSamp, int _totNodes, const int _minHeight, int &auxRvSize);
  static void ReFactory(int _nodeMax, int liveCount);
  static void DeFactory();
  static void TreeInit(int bagCount);
  void Begin(int id, int _idxCount, int _sCount, bool _isLH, double _sum, SplitNode *_par);
  static int NextLevel(int liveCount, int level);
  static void NextLevel(int countNext);
  static int Levels(int bagCount, const double auxRv[]);
  static void LeafComplete(Leaf *leaf, int idxCount, double sum, int sCount);
  static double GetPrebias(int liveIdx) {
    return node->Prebias(liveIdx);
  }
  virtual double Prebias(int liveIdx) = 0;
  virtual void CacheFields(int liveIdx, int &_idxCount, int &_sCount, double &_sum, double &_preBias, class SplitNode* &_par, bool &_isLH) = 0;
  virtual double SampleReplay(int predIdx, int level, int start, int lhCount, int id);
  virtual void Reset(int countNext, const int sample2Node[]) = 0;
  virtual void EarlyFields(int, int, int, bool, double, SplitNode*) = 0;
  virtual int LateFields(int, int) = 0;
  virtual void SetPrebias(int liveIdx) = 0;
};

// Regression response.
//
class NodeReg : public Node {
  Sample *sample;
  NodeReg *node;
  int *sample2Rank;
 public:
  static void Factory();
  static void ReFactory();
  static void DeFactory();
  static void TreeInit(int bagCount, const double auxRv[]);
  static void ClearTree();

  // Initializes most of the fields common to Nodes.  These are the fields with values
  // known either when splits are resolved or on entry to level zero (TreeInit).
  //
  inline void EarlyFields(int id, int _idxCount, int _sCount, bool _isLH, double _sum, SplitNode *_par) {
     NodeReg *nd = &node[id];
     nd->idxCount = _idxCount;
     nd->sCount = _sCount;
     nd->isLH = _isLH;
     nd->sum = _sum;
     nd->par = _par;
   }

  inline int LateFields(int id, int off) {
    NodeReg *nd = &node[id];
    nd->SetPrebias(id);
    nd->lhStart = off;

    return off + nd->idxCount;
  }

  inline double Prebias(int liveIdx) {
    return node[liveIdx].preBias;
  }

  inline void CacheFields(int liveIdx, int &_idxCount, int &_sCount, double &_sum, double &_preBias, class SplitNode* &_par, bool &_isLH) {
    NodeReg* nd = &node[liveIdx];
    _idxCount = nd->idxCount;
    _sCount = nd->sCount;
    _sum = nd->sum;
    _preBias = nd->preBias;
    _par = nd->par;
    _isLH = nd->isLH;
  }
  inline double SampleReplay(int predIdx, int level, int start, int lhCount, int id) {
    return PredReg::SampleReplay(predIdx, level, start, lhCount, id);
  }
  void Reset(int countNext, const int sample2Node[]) {}
  void SetPrebias(int liveIdx);
  void LeafComplete(Leaf *leaf, int idxCount, double sum, int sCount);
};

// Cateogorical response.
// Each response factor must be tracked per node.
//
class NodeCtg : public Node {
  NodeCtg *node;
  int ctgWidth;
  int *yCtg;
  SampleCtg *sampleCtg;
  double *ctgSum;
 public:
  double sumSquares;
  static double *ctgSum;
  static void Factory(int _yCtg[], int _ctgWidth, int &auxSize);
  static void ReFactory();
  static void DeFactory();
  static void ClearTree();
  static void TreeInit(int bagCount, const double auxRv[]);

  // Initializes most of the fields common to Nodes.  These are the fields with values
  // known either when splits are resolved or on entry to level zero (TreeInit).
  //
  inline void EarlyFields(int id, int _idxCount, int _sCount, bool _isLH, double _sum, SplitNode *_par) {
     NodeCtg *nd = &node[id];
     nd->idxCount = _idxCount;
     nd->sCount = _sCount;
     nd->isLH = _isLH;
     nd->sum = _sum;
     nd->par = _par;
   }

  inline int LateFields(int id, int off) {
    NodeCtg *nd = &node[id];
    nd->SetPrebias(id);
    nd->lhStart = off;

    return off + nd->idxCount;
  }

  inline void CacheFields(int liveIdx, int &_idxCount, int &_sCount, double &_sum, double &_preBias, class SplitNode* &_par, bool &_isLH) {
    NodeCtg nd = node[liveIdx];
    _idxCount = nd.idxCount;
    _sCount = nd.sCount;
    _sum = nd.sum;
    _preBias = nd.preBias;
    _par = nd.par;
    _isLH = nd.isLH;
  }
  inline double SampleReplay(int predIdx, int level, int start, int lhCount, int id) {
    return PredCtg::SampleReplay(predIdx, level, start, lhCount, id);
  }

  inline double Prebias(int liveIdx) {
    return node[liveIdx].preBias;
  }

  void SetPrebias(int liveIdx);
  void Reset(int countNext, const int sample2Node[]);
};


// Caches intermediate Node contents during intra-level transfer.
//
class NodeCache : public Node {
  NodeCache *nodeCache;
 public:
  static int liveCount;
  static int minHeight;
  int leafOrPred; // >=0: splitting predictor; < 0: leaf id.
  SplitNode *preTree; // Pre-tree node created for this accumulator.
  double lhSum; // Sum of LHS values.
  int lhSCount; // # samples on LHS.
  int lhIdxCount; // # index entries on LHS.
  int lhId; // Live offset of LHS at next level.
  int rhId; // Live offset of RHS at next level.
  static void CacheNodes(int liveCount);
  static int LHRH(int liveCount, int level);
  static void TransferLeaves(int liveCount);
  static void SetNode(int liveIdx, int _lhStart, int _idxCount, int _sCount, double _sum, double _preBias, SplitNode *_par, bool _isLH);
  static void LHRH1(int liveIdx, int level int &countNext);
  static double SampleReplayLHRH(int predIdx, int liveIdx, int level, int lhCount);
  static void LHRH2();
  static void LevelToLeaf(int liveCount, int bagCount, int sample2Node[]);
  static void Restage(int liveCount, int predIdx, int level);
  void Restage(int predIdx, int level);

  // Invoked from the RHS of a split to determine whether the node persists to the next
  // level.  Returns true if the node subsumes too few samples or is representable as a
  // single buffer element.
  //
  inline bool TerminalRight(const int _sCount, const int _lhICount) {
    return sCount - _sCount < minHeight || _lhICount >= idxCount - 1;
  }

  // Invoked from the LHS of a split to determine whether the node persists to the next
  // level.  Returns true if the node subsumes too few samples or if the node is
  // representable as a single buffer element.
  static inline bool TerminalLeft(const int _sCount, const int _lhICount) {
    return _sCount < minHeight || _lhICount == 1;
  }
};

#endif
