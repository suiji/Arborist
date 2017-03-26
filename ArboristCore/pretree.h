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
#include <algorithm>

#include "param.h"


/**
   @brief Key for translating terminal vector.
 */
class TermKey {
 public:
  unsigned int base;
  unsigned int extent;
  unsigned int ptId;

  inline void Init(unsigned int _base, unsigned int _extent, unsigned int _ptId) {
    base = _base;
    extent = _extent;
    ptId = _ptId;
  }
};



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
  unsigned int lhId;  // LH subnode index. Nonzero iff non-terminal.
  unsigned int predIdx; // Split only.
  union {
    unsigned int offset; // Bit-vector offset:  factor.
    RankRange rankRange;//double rkMean; // Mean rank:  numeric.
  } splitVal;
  void Consume(const class PMTrain *pmTrain, class ForestTrain *forest, unsigned int tIdx);
};


class PreTree {
  static unsigned int heightEst;
  const class PMTrain *pmTrain;
  PTNode *nodeVec; // Vector of tree nodes.
  unsigned int nodeCount; // Allocation height of node vector.
  unsigned int height;
  unsigned int leafCount;
  unsigned int bitEnd; // Next free slot in factor bit vector.
  class BV *splitBits;
  std::vector<TermKey> termKey;
  std::vector<unsigned int> termST;
  class BV *BitFactory();
  void TerminalOffspring(unsigned int _parId);
  const std::vector<unsigned int> FrontierToLeaf(class ForestTrain *forest, unsigned int tIdx);
  const unsigned int bagCount;
  std::vector<double> info; // Aggregates info value of nonterminals, by predictor.
  unsigned int BitWidth();

 public:
  PreTree(const class PMTrain *_pmTrain, unsigned int _bagCount);
  ~PreTree();
  static void Immutables(unsigned int _nSamp, unsigned int _minH);
  static void DeImmutables();
  static void Reserve(unsigned int height);

  const std::vector<unsigned int> DecTree(class ForestTrain *forest, unsigned int tIdx, std::vector<double> &predInfo);
  void NodeConsume(class ForestTrain *forest, unsigned int tIdx);
  void BitConsume(unsigned int *outBits);
  void LHBit(int idx, unsigned int pos);
  void NonTerminalFac(double _info, unsigned int _predIdx, unsigned int _id);
  void NonTerminalNum(double _info, unsigned int _predIdx, RankRange _rankRange, unsigned int _id);
  void Level(unsigned int splitNext, unsigned int leafNext);
  void ReNodes();
  void SubtreeFrontier(const std::vector<TermKey> &stKey, const std::vector<unsigned int> &stTerm);

  
  /**
     @brief Height accessor.
   */
  inline unsigned int Height() {
    return height;
  }


  inline unsigned int LHId(unsigned int ptId) const {
    return nodeVec[ptId].lhId;
  }

  
  inline unsigned int RHId(unsigned int ptId) const {
    unsigned int lhId = nodeVec[ptId].lhId;
    return lhId != 0 ? lhId + 1 : 0;
  }

  
  /**
     @return true iff node is nonterminal.
   */
  inline bool NonTerminal(unsigned int ptId) {
    return nodeVec[ptId].lhId > 0;
  }

  
  /**
     @brief Fills in references to values known to be useful for building
     a block of PreTree objects.

     @return void.
   */
  inline void BlockBump(unsigned int &_height, unsigned int &_maxHeight, unsigned int &_bitWidth, unsigned int &_leafCount, unsigned int &_bagCount) {
    _height += height;
    _maxHeight = std::max(height, _maxHeight);
    _bitWidth += BitWidth();
    _leafCount += leafCount;
    _bagCount += bagCount;
  }
};

#endif
