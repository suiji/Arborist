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

#ifndef PARTITION_PRETREE_H
#define PARTITION_PRETREE_H

#include <vector>
#include <algorithm>

#include "ptnode.h" // Algorithm-dependent definition.
#include "decnode.h" 

/**
 @brief Serialized representation of the pre-tree, suitable for tranfer between
 devices such as coprocessors, disks and nodes.
*/
class PreTree {
  static size_t heightEst;
  static size_t leafMax; // User option:  maximum # leaves, if > 0.
  const unsigned int bagCount;
  size_t height;
  size_t leafCount;
  size_t bitEnd; // Next free slot in factor bit vector.
  vector<class PTNode> nodeVec; // Vector of tree nodes.
  vector<struct SplitCrit> splitCrit;
  class BV *splitBits;
  vector<unsigned int> termST;

  /**
     @brief Constructs mapping from sample indices to leaf indices.

     @param[in, out] forest accumulates the growing forest.

     @return rewritten map.
  */
  const vector<unsigned int> frontierConsume(class ForestTrain *forest) const;


  /**
     @return BV-aligned length of used portion of split vector.
  */
  IndexT getBitWidth();


  /**
     @brief Accounts for the addition of two terminals to the tree.

     @return void, with incremented height and leaf count.
  */
  inline void terminalOffspring() {
  // Two more leaves for offspring, one fewer for this.
    height += 2;
    leafCount++;
  }


 public:
  PreTree(const class SummaryFrame* frame,
          const class Frontier* frontier);

  ~PreTree();
  static void immutables(size_t _nSamp, size_t _minH, size_t _leafMax);
  static void deImmutables();


  /**
     @brief Refines the height estimate using the actual height of a
     constructed PreTree.

     @param height is an actual height value.
  */
  static void reserve(size_t height);


  /**
     @brief Dispatches nonterminal method according to split type.
   */
  void nonterminal(double info,
                   class IndexSet* iSet);

  
  /**
     @brief Appends criterion for bit-based branch.

     @param predIdx is the criterion predictor.

     @param cardinality is the predictor's cardinality.
  */
  void critBits(const class IndexSet* iSet,
                unsigned int predIdx,
                unsigned int cardinality);

  /**
     @brief Appends criterion for cut-based branch.
     
     @param rankRange bounds the cut-defining ranks.
  */
  void critCut(const class IndexSet* iSet,
               unsigned int predIdx,
               const IndexRange& rankRange);


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
  /**
     @brief Consumes nonterminal information into the dual-use vectors needed by the decision tree.

     Leaf information is post-assigned by the response-dependent Sample methods.

     @param[in, out]  forest inputs/outputs the updated forest.

     @param[out] predInfo outputs the predictor-specific information values.
  */
  void consumeNonterminal(class ForestTrain *forest,
                          vector<double> &predInfo) const;

  void BitConsume(unsigned int *outBits);


  /**
     @brief Sets specified bit in (left) splitting bit vector.

     @param iSet is the index node for which the LH bit is set.

     @param pos is the bit position beyond to set.
  */
  void setLeft(const class IndexSet* iSet,
               IndexT pos);


  /**
     @brief Absorbs the terminal list from a completed subtree.

     Side-effects the frontier map.

     @param stTerm are subtree-relative indices.  These must be mapped to
     sample indices if the subtree is proper.
  */
  void subtreeFrontier(const vector<unsigned int> &stTerm);

  
  unsigned int LeafMerge();

  
  inline IndexT getLHId(IndexT ptId) const {
    return nodeVec[ptId].getLHId(ptId);
  }

  
  inline IndexT getRHId(IndexT ptId) const {
    return nodeVec[ptId].getRHId(ptId);
  }


  inline IndexT getSuccId(IndexT ptId, bool isLeft) const {
    return isLeft ? nodeVec[ptId].getLHId(ptId) : nodeVec[ptId].getRHId(ptId);
  }
  
  /**
     @return true iff node is nonterminal.
   */
  inline bool isNonTerminal(IndexT ptId) const {
    return nodeVec[ptId].isNonTerminal();
  }


    /**
       @brief Determines whether a nonterminal can be merged with its
       children.

       @param ptId is the index of a nonterminal.

       @return true iff node has two leaf children.
    */
  inline bool isMergeable(IndexT ptId) const {
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
    _bitWidth += getBitWidth();
    _leafCount += leafCount;
    _bagCount += bagCount;
  }
};

#endif
