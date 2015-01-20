// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#ifndef ARBORIST_PRETREE_H
#define ARBORIST_PRETREE_H

// Serialized representation of the pre-tree, suitable for tranfer between
// devices such as coprocessors, disks and nodes.  Left and right subnodes
// are referenced as indices into the vector representation of the tree.
// Leaves are distinguished as having two negative-valued subnode indices,
// while splits have both subset indices positive.  Mixed negative and non-
// negative subnode indices indicate an error.
// 
class PreTree {
  int lhId;  // LH subnode index. Non-negative iff non-terminal.
  int predIdx; // Split only.
  double splitVal; // Split only.
  double info; // Split only.
  static int ptCount; // Allocation height of preTree[].
  static int bitLength; // Length of bit vector recording factor-valued splits.
  static PreTree *preTree;
  static int levelMax;
  static int levelOffset;
  static int treeHeight;
  static int levelBase;
  static int leafCount;
  static int splitCount;
  static int treeBitOffset;
  static bool *treeSplitBits;
  static int *qOff;
  static int *qRanks;
  //  static int rowBlock;  // Coprocessor stride.
  static bool IsNT(int idx) {
    return preTree[idx].lhId > 0;
  }
  static int *sample2PT; // Needs to be shared with SampleReg methods.

 public:
  static int Sample2Leaf(int sIdx);

  static int TreeBitOffset() {
    return treeBitOffset;
  }

  static bool BitVal(int pos) {
    return treeSplitBits[pos];
  }

  static inline int Sample2PT(int sIdx) {
    return sample2PT[sIdx];
  }

  static inline void MapSample(int sIdx, int id) {
    sample2PT[sIdx] = id;
  }
  static inline int TreeHeight() {
    return treeHeight;
  }

  // Updates level base to current tree height.
  //
  static void NextLevel() {
    levelBase = treeHeight;
  }

  // Returns the level-relative offset of tree index 'ptId'.
  //
  static int LevelOff(int ptId) {
    return ptId - levelBase;
  }

  static int LevelSampleOff(int sIdx) {
    return Sample2PT(sIdx) - levelBase;
  }

  static int LevelWidth() {
    return treeHeight - levelBase;
  }

// After all SplitSigs in the tree are lowered, returns the total width of factors
// seen as splitting values.
//
  static int SplitFacWidth() {
    return treeBitOffset;
  }

  // Allocates the bit string for the current (pre)tree and initializes to
  // false.
  //
  static inline bool *BitFactory(int length) {
    bool *tsb = new bool[length];
    for (int i = 0; i < length; i++)
      tsb[i] = false;

    return tsb;
  }

  static void TreeInit(int _levelMax, int _bagCount);
  static void TreeClear();
  static void TerminalOffspring(int _parId, int &ptLH, int &ptRH);
  static void SingleBit(int pos);
  static void NonTerminalFac(int treeId, double info, int predIdx);
  static void NonTerminalGeneric(int _id, double _info, double _splitVal, int _predIdx);
  static void CheckStorage(int splitNext, int leafNext);
  static void ReBits();
  static void ReFactory();
  static double FacBits(const bool facBits[], int facWidth);
  static void ConsumeNodes(int leafPred, int predVec[], double splitVec[], int bumpVec[], double scoreVec[]);
  static void ConsumeSplitBits(int outBits[]);
};

#endif
