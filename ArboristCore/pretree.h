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
  unsigned int lhDel;  // Delta to LH subnode. Nonzero iff non-terminal.
  unsigned int predIdx; // Nonterminal only.
  FltVal info; // Nonterminal only.
  union {
    unsigned int offset; // Bit-vector offset:  factor.
    RankRange rankRange; // LH, RH ranks:  numeric.
  } splitVal;

  void NonterminalConsume(const class PMTrain *pmTrain, class ForestTrain *forest, unsigned int tIdx, std::vector<double> &predInfo, unsigned int idx) const;


  inline void SetTerminal() {
    lhDel = 0;
  }


  inline void SetNonterminal(unsigned int parId, unsigned int lhId) {
    lhDel = lhId - parId;
  }

  
  inline bool NonTerminal() const {
    return lhDel != 0;
  }


  inline unsigned int LHId(unsigned int ptId) const {
    return NonTerminal() ? ptId + lhDel : 0;
  }

  inline unsigned int RHId(unsigned int ptId) const {
    return NonTerminal() ? LHId(ptId) + 1 : 0;
  }
};


class PreTree {
  static unsigned int heightEst;
  static unsigned int leafMax; // User option:  maximum # leaves, if > 0.
  const class PMTrain *pmTrain;
  PTNode *nodeVec; // Vector of tree nodes.
  unsigned int nodeCount; // Allocation height of node vector.
  unsigned int height;
  unsigned int leafCount;
  unsigned int bitEnd; // Next free slot in factor bit vector.
  class BV *splitBits;
  std::vector<unsigned int> termST;
  class BV *BitFactory();
  void TerminalOffspring(unsigned int _parId);
  const std::vector<unsigned int> FrontierConsume(class ForestTrain *forest, unsigned int tIdx) const ;
  const unsigned int bagCount;
  unsigned int BitWidth();

 public:
  PreTree(const class PMTrain *_pmTrain, unsigned int _bagCount);
  ~PreTree();
  static void Immutables(unsigned int _nSamp, unsigned int _minH, unsigned int _leafMax);
  static void DeImmutables();
  static void Reserve(unsigned int height);

  const std::vector<unsigned int> Consume(class ForestTrain *forest, unsigned int tIdx, std::vector<double> &predInfo);
  void NonterminalConsume(class ForestTrain *forest, unsigned int tIdx, std::vector<double> &predInfo) const;
  void BitConsume(unsigned int *outBits);
  void LHBit(int idx, unsigned int pos);
  void NonTerminalFac(double _info, unsigned int _predIdx, unsigned int _id);
  void NonTerminalNum(double _info, unsigned int _predIdx, RankRange _rankRange, unsigned int _id);
  void Level(unsigned int splitNext, unsigned int leafNext);
  void ReNodes();
  void SubtreeFrontier(const std::vector<unsigned int> &stTerm);
  unsigned int LeafMerge();
  
  inline unsigned int LHId(unsigned int ptId) const {
    return nodeVec[ptId].LHId(ptId);
  }

  
  inline unsigned int RHId(unsigned int ptId) const {
    return nodeVec[ptId].RHId(ptId);
  }

  
  /**
     @return true iff node is nonterminal.
   */
  inline bool NonTerminal(unsigned int ptId) const {
    return nodeVec[ptId].NonTerminal();
  }


    /**
       @brief Determines whether a nonterminal can be merged with its
       children.

       @param ptId is the index of a nonterminal.

       @return true iff node has two leaf children.
    */
  inline bool Mergeable(unsigned int ptId) const {
    return !NonTerminal(LHId(ptId)) && !NonTerminal(RHId(ptId));
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
