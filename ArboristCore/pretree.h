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

#include <vector>

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
  unsigned int id;
  unsigned int lhId;  // LH subnode index. Positive iff non-terminal.
  unsigned int predIdx; // Split only.
  union {
    unsigned int offset; // Bit-vector offset:  factor.
    double rkMean; // Mean rank:  numeric.
  } splitVal;
  void Consume(class Forest *forest, unsigned int tIdx);
};


class PreTree {
  static unsigned int nPred;
  static unsigned int heightEst;
  PTNode *nodeVec; // Vector of tree nodes.
  int nodeCount; // Allocation height of node vector.
  int height;
  unsigned int leafCount;
  unsigned int bitEnd; // Next free slot in factor bit vector.
  unsigned int *sample2PT; // Public accessor is Sample2Frontier().
  double *info; // Aggregates info value of nonterminals, by predictor.
  class BV *splitBits;
  class BV *BitFactory();
  void TerminalOffspring(unsigned int _parId, unsigned int &ptLH, unsigned int &ptRH);
  const std::vector<unsigned int> FrontierToLeaf(class Forest *forest, unsigned int tIdx);
  unsigned int bagCount;

 public:
  PreTree(unsigned int _bagCount);
  ~PreTree();
  static void Immutables(unsigned int _nPred, unsigned int _nSamp, unsigned int _minH);
  static void DeImmutables();
  static void Reserve(unsigned int height);

  const std::vector<unsigned int> DecTree(class Forest *forest, unsigned int tIdx, double predInfo[]);
  void NodeConsume(class Forest *forest, unsigned int tIdx);
  unsigned int BitWidth();
  void BitConsume(unsigned int *outBits);

  
  /**
   @brief Maps sample index to index of frontier node with which it is currently associated.
 
   @param sIdx is the index of a sample

   @return pretree index.
  */
  inline unsigned int Sample2Frontier(int sIdx) const {
    return sample2PT[sIdx];
  }


  inline unsigned int LeafCount() const {
    return leafCount;
  }
  

  inline int Height() const {
    return height;
  }

  
  inline int BagCount() const {
    return bagCount;
  }

  void LHBit(int idx, unsigned int pos);
  void NonTerminalFac(double _info, unsigned int _predIdx, unsigned int _id, unsigned int &ptLH, unsigned int &ptRH);
  void NonTerminalNum(double _info, unsigned int _predIdx, unsigned int _rkLow, unsigned int _rkHigh, unsigned int _id, unsigned int &ptLH, unsigned int &ptRH);

  double Replay(class SamplePred *samplePred, unsigned int predIdx, unsigned int targBit, int start, int end, unsigned int ptId);
  
  void CheckStorage(int splitNext, int leafNext);
  void ReNodes();
};

#endif
