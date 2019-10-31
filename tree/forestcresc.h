// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file forestcresc.h

   @brief Data structures and methods for growing the decision forest.

   @author Mark Seligman
 */

#ifndef TREE_FORESTCRESC_H
#define TREE_FORESTCRESC_H

#include <vector>

#include "fbcresc.h"



/**
   @brief struct CartNode block for crescent frame;
 */
template<typename treeType>
class NBCresc {
  vector<treeType> treeNode;
  vector<size_t> height;
  size_t treeFloor; // Block-relative index of current tree floor.

public:
  /**
     @brief Constructor.

     @param treeChunk is the number of trees in the current block.
   */
  NBCresc(unsigned int treeChunk) :
    treeNode(vector<treeType>(0)),
    height(vector<size_t>(treeChunk)) {
  }


  /**
     @brief Allocates and initializes nodes for current tree.

     @param tIdx is the block-relative tree index.

     @param nodeCount is the number of tree nodes.
   */
  void treeInit(unsigned int tIdx,
                unsigned int nodeCount) {
    treeFloor = treeNode.size();
    height[tIdx] = treeFloor + nodeCount;
    treeType tn;
    treeNode.insert(treeNode.end(), nodeCount, tn);
  }



  /**
     @brief Copies treeNode contents by byte.

     @param[out] nodeRaw outputs the raw contents.
   */
  void dumpRaw(unsigned char nodeRaw[]) const {
    for (size_t i = 0; i < treeNode.size() * sizeof(treeType); i++) {
      nodeRaw[i] = ((unsigned char*) &treeNode[0])[i];
    }
  }


  /**
     @brief Tree-level dispatch to low-level member.

     Parameters as with low-level implementation.
  */
  void splitUpdate(const class SummaryFrame* sf) {
    for (auto & tn : treeNode) {
      tn.setQuantRank(sf);
    }
  }


  
  /**
     @brief Accessor for height vector.
   */
  const vector<size_t>& getHeight() const {
    return height;
  }


  /**
     @brief Sets looked-up nonterminal node to values passed.

     @param nodeIdx is a tree-relative node index.

     @param decNode contains the value to set.

     @param isFactor is true iff the splitting predictor is categorical.
  */
  void branchProduce(unsigned int nodeIdx,
		     IndexT lhDel,
		     const struct Crit& crit) {
    treeNode[treeFloor + nodeIdx].setBranch(lhDel, crit);
  }


  /**
    @brief Sets looked-up leaf node to leaf index passed.

    @return void.

  */
  void leafProduce(unsigned int nodeIdx,
		   unsigned int leafIdx) {
    treeNode[treeFloor + nodeIdx].setLeaf(leafIdx);
  }
};


/**
   @brief Class definition for crescent forest.
 */
template<typename treeType>
class ForestCresc {
  unique_ptr<NBCresc<treeType> > nbCresc; // Crescent node block.
  unique_ptr<FBCresc> fbCresc; // Crescent factor-summary block.

 public:
  /**
     @brief Constructs block of trees for crescent forest, wrapping
     vectors allocated by the front-end bridge.

     @param treeChunk is the number of trees to train.
   */
  ForestCresc(unsigned int treeChunk) :
    nbCresc(make_unique<NBCresc<treeType>>(treeChunk)),
    fbCresc(make_unique<FBCresc>(treeChunk)) {
  }

  
  ~ForestCresc() {
  }

  
  /**
     @brief Wrapper for bit vector appending.

     @param splitBits encodes bits maintained for the current tree.

     @param bitEnd is the final referenced bit position.

     @param tIdx is the tree index.
   */
  void appendBits(const class BV *splitBits,
                  size_t bitEnd,
                  unsigned int tIdx) {
    fbCresc->appendBits(splitBits, bitEnd, tIdx);
  }

  
  /**
     @brief Allocates and initializes sufficient nodes for current tree.

     @param tIdx is the block-relative tree index.

     @param nodeCount is the number of nodes.
   */
  void treeInit(unsigned int tIdx,
                unsigned int nodeCount) {
    nbCresc->treeInit(tIdx, nodeCount);
  }


  /**
     @brief Precipitates production of a branch node in the crescent forest.

     @param frame summarizes the training observations.

     @param idx is a tree-relative node index.

     @parm decNode contains the value to set.
   */
  void nonTerminal(IndexT nodeIdx,
                   IndexT lhDel,
                   const struct Crit& crit) {
    nbCresc->branchProduce(nodeIdx, lhDel, crit);
  }



  /**
     @brief Outputs raw byes of node vector.

     @return void.
   */
  void cacheNodeRaw(unsigned char rawOut[]) const {
    nbCresc->dumpRaw(rawOut);
  }

  /**
     @brief Dumps raw splitting values for factors.

     @param[out] rawOut outputs the raw bytes of factor split values.
   */
  void cacheFacRaw(unsigned char rawOut[]) const {
    fbCresc->dumpRaw(rawOut);
  }


  /**
     @brief Accessor for vector of tree heights.

     @return reference to tree-height vector.
   */
  const vector<size_t>& getNodeHeight() const {
    return nbCresc->getHeight();
  }
  

  /**
     @brief Accessor for vector of factor-split heights.

     @return reference to factor-height vector.
   */
  const vector<size_t>& getFacHeight() const {
    return fbCresc->getHeight();
  }


  /**
    @brief Sets tree node as terminal.

    @param nodeIdx is a tree-relative node index.

    @param leafIdx is a tree-relative leaf index.
  */
  void terminal(unsigned int nodeIdx,
                unsigned int leafIdx) {
    nbCresc->leafProduce(nodeIdx, leafIdx);
  }

  
  /**
     @brief Post-pass to update numerical splitting values from ranks.

     @param summaryFrame records the predictor types.
  */
  void splitUpdate(const class SummaryFrame* sf) {
    nbCresc->splitUpdate(sf);
  }
};

#endif
