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

#include "decnode.h"


/**
  @brief DecNode specialized for training.
 */
class PTNode : public DecNode {
  FltVal info;  // Nonzero iff nonterminal.
 public:
  
  void consumeNonterminal(const class FrameTrain *frameTrain,
                          class ForestTrain *forest,
                          vector<double> &predInfo,
                          unsigned int idx) const;

  void splitNum(const class SplitCand &cand,
                unsigned int lhDel);

  /**
     @brief Resets to default terminal status.

     @return void.
   */
  inline void setTerminal() {
    lhDel = 0;
  }


  /**
     @brief Resets to nonterminal with specified lh-delta.

     @return void.
   */
  inline void setNonterminal(unsigned int lhDel) {
    this->lhDel = lhDel;
  }

  
  inline bool isNonTerminal() const {
    return lhDel != 0;
  }


  inline unsigned int getLHId(unsigned int ptId) const {
    return isNonTerminal() ? ptId + lhDel : 0;
  }

  inline unsigned int getRHId(unsigned int ptId) const {
    return isNonTerminal() ? getLHId(ptId) + 1 : 0;
  }


  inline void SplitFac(unsigned int predIdx, unsigned int lhDel, unsigned int bitEnd, double info) {
    this->predIdx = predIdx;
    this->lhDel = lhDel;
    this->splitVal.offset = bitEnd;
    this->info = info;
  }
};


/**
 @brief Serialized representation of the pre-tree, suitable for tranfer between
 devices such as coprocessors, disks and nodes.
*/
class PreTree {
  static size_t heightEst;
  static size_t leafMax; // User option:  maximum # leaves, if > 0.
  const class FrameTrain *frameTrain;
  const unsigned int bagCount;
  size_t nodeCount; // Allocation height of node vector.
  PTNode *nodeVec; // Vector of tree nodes.
  size_t height;
  size_t leafCount;
  size_t bitEnd; // Next free slot in factor bit vector.
  class BV *splitBits;
  vector<unsigned int> termST;
  class BV *BitFactory();
  const vector<unsigned int> frontierConsume(class ForestTrain *forest) const;
  unsigned int BitWidth();


  /**
     @brief Accounts for the addition of two terminals to the tree.

     @return void, with incremented height and leaf count.
  */
  inline void TerminalOffspring() {
  // Two more leaves for offspring, one fewer for this.
    height += 2;
    leafCount++;
  }


 public:
  PreTree(const class FrameTrain *_frameTrain, unsigned int _bagCount);
  ~PreTree();
  static void immutables(size_t _nSamp, size_t _minH, size_t _leafMax);
  static void deImmutables();
  static void reserve(size_t height);


  /**
     @brief Consumes all pretree nonterminal information into crescent forest.

     @param forest grows by producing nodes and splits consumed from pre-tree.

     @param tIdx is the index of the tree being consumed/produced.

     @param predInfo accumulates the information contribution of each predictor.

     @return leaf map from consumed frontier.
  */
  const vector<unsigned int> consume(class ForestTrain *forest,
                                     unsigned int tIdx,
                                     vector<double> &predInfo);

  void consumeNonterminal(class ForestTrain *forest,
                          vector<double> &predInfo) const;

  void BitConsume(unsigned int *outBits);

  void LHBit(int idx, unsigned int pos);

  void branchFac(const class SplitCand& argMax,
                 unsigned int _id);

  /**
     @brief Finalizes numeric-valued nonterminal.

     @param argMax is the split candidate characterizing the branch.

     @param id is the node index.
  */
  void branchNum(const class SplitCand &argMax,
                 unsigned int id);

  void levelStorage(unsigned int splitNext, unsigned int leafNext);
  void ReNodes();
  void subtreeFrontier(const vector<unsigned int> &stTerm);
  unsigned int LeafMerge();
  
  inline unsigned int getLHId(unsigned int ptId) const {
    return nodeVec[ptId].getLHId(ptId);
  }

  
  inline unsigned int getRHId(unsigned int ptId) const {
    return nodeVec[ptId].getRHId(ptId);
  }

  
  /**
     @return true iff node is nonterminal.
   */
  inline bool isNonTerminal(unsigned int ptId) const {
    return nodeVec[ptId].isNonTerminal();
  }


    /**
       @brief Determines whether a nonterminal can be merged with its
       children.

       @param ptId is the index of a nonterminal.

       @return true iff node has two leaf children.
    */
  inline bool isMergeable(unsigned int ptId) const {
    return !isNonTerminal(getLHId(ptId)) && !isNonTerminal(getRHId(ptId));
  }  

  
  /**
     @brief Fills in references to values known to be useful for building
     a block of PreTree objects.

     @return void.
   */
  inline void blockBump(size_t &_height,
                        size_t &_maxHeight,
                        size_t &_bitWidth,
                        size_t &_leafCount,
                        size_t &_bagCount) {
    _height += height;
    _maxHeight = max(height, _maxHeight);
    _bitWidth += BitWidth();
    _leafCount += leafCount;
    _bagCount += bagCount;
  }
};

#endif
