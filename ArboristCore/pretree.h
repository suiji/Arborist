// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file pretree.h

   @brief Class defintions for the pre-tree, a serial and minimal representation from which the decision tree is built.

   @author Mark Seligman

 */

#ifndef ARBORIST_PRETREE_H
#define ARBORIST_PRETREE_H

/**
 @brief Serialized representation of the pre-tree, suitable for tranfer between
 devices such as coprocessors, disks and nodes.

 Left and right subnodes are referenced as indices into the vector
 representation of the tree. Leaves are distinguished as having two
 negative-valued subnode indices, while splits have both subset
 indices positive.  Mixed negative and non-negative subnode indices
 indicate an error.
*/
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

  /**
     @brief Indicates whether node is terminal.

     @param idx is the node index.

     @retrun true iff node is nonterminal.
   */
  static bool IsNT(int idx) {
    return preTree[idx].lhId > 0;
  }
  static int *sample2PT; // Needs to be shared with SampleReg methods.

 public:
  static int Sample2Leaf(int sIdx);

  /**
     @return offset into the split-value bit vector for the current level.
   */
  static int TreeBitOffset() {
    return treeBitOffset;
  }

  /**
     @return true iff bit at position 'pos' is set.
   */
  static bool BitVal(int pos) {
    return treeSplitBits[pos];
  }

  /**
     @brief Maps sample index into pretree node.

     @param sIdx is the index of a sample.

     @return pretree index of node currently referencing the index.
   */
  static inline int Sample2PT(int sIdx) {
    return sample2PT[sIdx];
  }

  /**
     @brief Writes the sample map.

     @param sIdx is a sample index.

     @param id is a pretree index.

     @return void.
   */
  static inline void MapSample(int sIdx, int id) {
    sample2PT[sIdx] = id;
  }

  /**
     @return current pretree height.
  */
  static inline int TreeHeight() {
    return treeHeight;
  }

  /**
    @brief Updates level base to current tree height.

    @return void.
  */
  static void NextLevel() {
    levelBase = treeHeight;
  }

  /**
    @return the level-relative offset of tree index 'ptId'.
  */
  static int LevelOff(int ptId) {
    return ptId - levelBase;
  }

  /**
     @brief Same as above, but with sample index argument.
   */
  static int LevelSampleOff(int sIdx) {
    return Sample2PT(sIdx) - levelBase;
  }

  /**
     @return count of pretree nodes at current level.
  */
  static int LevelWidth() {
    return treeHeight - levelBase;
  }

  /**
     @return  total accumulated width of factors seen as splitting values.
  */
  static int SplitFacWidth() {
    return treeBitOffset;
  }

  /**
     @brief Allocates the bit string for the current (pre)tree and initializes to false.
     @param length is the length of the bit vector.

     @return void.
  */
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
