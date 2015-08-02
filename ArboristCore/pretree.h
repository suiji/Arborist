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
class PTNode {
 public:
  int id;
  int lhId;  // LH subnode index. Non-negative iff non-terminal.
  int predIdx; // Split only.
  double splitVal; // Split only.
  double info; // Split only.
  void Consume(int &pred, double &num, int &bump);
};


class PreTree {
  static unsigned int nRow;
  static unsigned int heightEst;
  class Sample *sample;
  PTNode *nodeVec; // Vector of tree nodes.
  int nodeCount; // Allocation height of node vector.
  int bitLength; // Length of bit vector recording factor-valued splits.
  int treeHeight;
  int leafCount;
  int treeBitOffset;
  int *sample2PT; // Public accessor is Sample2Frontier().
  unsigned int *inBag;
  bool *treeSplitBits;
  inline bool *BitFactory(int bitLength = 0);
 protected:
  int bagCount;
  /**
     @brief Indicates whether node is terminal.

     @param idx is the node index.

     @retrun true iff node is nonterminal.
   */
  bool IsNT(int ptIdx) {
    return nodeVec[ptIdx].lhId > 0;
  }

 public:
  PreTree();
  ~PreTree();
  static void Immutables(unsigned int _nRow, unsigned int _nSamp, unsigned int _minH);
  static void DeImmutables();
  static void RefineHeight(unsigned int height);
  
  class SplitPred *BagRows(const class PredOrd *predOrd, class SamplePred *samplePred, int &_bagCount, double &_sum);

  /**
     @return offset into the split-value bit vector for the current level.
   */
  int TreeBitOffset() {
    return treeBitOffset;
  }


  /**
     @return true iff bit at position 'pos' is set.
   */
  bool BitVal(int pos) {
    return treeSplitBits[pos];
  }

  /**
   @brief Maps sample index to index of frontier node with which it is currently associated.
 
   @param sIdx is the index of a sample

   @return pretree index.
  */
  inline int Sample2Frontier(int sIdx) const {
    return sample2PT[sIdx];
  }


  /**
     @return  total accumulated width of factors seen as splitting values.
  */
  inline int SplitFacWidth() const {
    return treeBitOffset;
  }

  
  inline int TreeHeight() const {
    return treeHeight;
  }

  
  inline int BagCount() const {
    return bagCount;
  }

  
  inline unsigned int *InBag() {
    return inBag;
  }

  void TerminalOffspring(int _parId, int &ptLH, int &ptRH);
  
  /**
     @brief Sets specified bit value to true.  Assumes initialized false.

     @param pos is the bit position beyond to set.

     @return void.
  */
  inline void LHBit(int pos) {
    treeSplitBits[treeBitOffset + pos] = true;
  }

  void NonTerminal(int _id, double _info, double _splitVal, int _predIdx);


  /**
     @brief Post-increments bit offset value.

     @param bump is the increment amount.

     @return cached bit offset value.
   */
  int PostBump(int bump) {
    int preVal = treeBitOffset;
    treeBitOffset += bump;

    return preVal;
  }

  double Replay(class SamplePred *samplePred, int predIdx, int level, int start, int end, int ptId);
  
  void CheckStorage(int splitNext, int leafNext);
  void ReBits();
  void ReNodes();
  double FacBits(const bool facBits[], int facWidth);
  void ConsumeNodes(int predVec[], double splitVec[], int bumpVec[]);
  void ConsumeSplitBits(int outBits[]);
  int LeafFields(int sIdx, int &sCount, unsigned int &rank) const;
};

#endif
