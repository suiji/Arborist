// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file foresttrain.h

   @brief Data structures and methods for training the decision forest.

   @author Mark Seligman
 */

#ifndef CART_FORESTTRAIN_H
#define CART_FORESTTRAIN_H

#include <vector>

#include "typeparam.h"


/**
   @brief struct CartNode block for crescent frame;
 */
class NBCresc {
  vector<struct CartNode> treeNode;
  vector<size_t> height;
  size_t treeFloor; // Block-relative index of current tree floor.

public:
  /**
     @brief Constructor.

     @param treeChunk is the number of trees in the current block.
   */
  NBCresc(unsigned int treeChunk);

  /**
     @brief Allocates and initializes nodes for current tree.

     @param tIdx is the block-relative tree index.

     @param nodeCount is the number of tree nodes.
   */
  void treeInit(unsigned int tIdx,
                unsigned int nodeCount);

  /**
     @brief Copies treeNode contents by byte.

     @param[out] nodeRaw outputs the raw contents.
   */
  void dumpRaw(unsigned char nodeRaw[]) const;


  /**
     @brief Tree-level dispatch to low-level member.

     Parameters as with low-level implementation.
  */
  void splitUpdate(const class SummaryFrame* sf);

  
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
		     const struct Crit& crit);


  /**
    @brief Sets looked-up leaf node to leaf index passed.

    @return void.

  */
  void leafProduce(unsigned int nodeIdx,
		   unsigned int leafIdx);
};


/**
   @brief Manages the crescent factor blocks.
 */
class FBCresc {
  vector<unsigned int> fac;  // Factor-encoding bit vector.
  vector<size_t> height; // Cumulative vector heights, per tree.

public:

  FBCresc(unsigned int treeChunk);

  /**
     @brief Sets the height of the current tree, storage now being known.

     @param tIdx is the tree index.
   */
  void treeCap(unsigned int tIdx);

  /**
     @brief Consumes factor bit vector and notes height.

     @param splitBits is the bit vector.

     @param bitEnd is the final bit position referenced.

     @param tIdx is the current tree index.
   */
  void appendBits(const class BV* splitBIts,
                  size_t bitEnd,
                  unsigned int tIdx);


  /**
     @brief Computes unit size for cross-compatibility of serialization.
   */
  static constexpr size_t unitSize() {
    return sizeof(unsigned int);
  }
  
  /**
     @brief Dumps factor bits as raw data.

     @param[out] facRaw outputs the raw factor data.
   */
  void dumpRaw(unsigned char facRaw[]) const;
  
  /**
     @brief Accessor for the per-tree height vector.

     @return reference to height vector.
   */
  const vector<size_t>& getHeight() const {
    return height;
  }
};


/**
   @brief Class definition for crescent forest.
 */
class ForestTrain {
  unique_ptr<NBCresc> nbCresc; // Crescent node block.
  unique_ptr<FBCresc> fbCresc; // Crescent factor-summary block.

 public:
  /**
     @brief Constructs block of trees for crescent forest, wrapping
     vectors allocated by the front-end bridge.

     @param treeChunk is the number of trees to train.
   */
  ForestTrain(unsigned int treeChunk);

  
  ~ForestTrain();

  
  /**
     @brief Wrapper for bit vector appending.

     @param splitBits encodes bits maintained for the current tree.

     @param bitEnd is the final referenced bit position.

     @param tIdx is the tree index.
   */
  void appendBits(const class BV *splitBits,
                  size_t bitEnd,
                  unsigned int tIdx);

  /**
     @brief Allocates and initializes sufficient nodes for current tree.

     @param tIdx is the block-relative tree index.

     @param nodeCount is the number of nodes.
   */
  void treeInit(unsigned int tIdx,
                unsigned int nodeCount);

  /**
     @brief Precipitates production of a branch node in the crescent forest.

     @param frame summarizes the training observations.

     @param idx is a tree-relative node index.

     @parm decNode contains the value to set.
   */
  void nonTerminal(IndexT idx,
                   IndexT lhDel,
                   const struct Crit& crit);

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
                unsigned int leafIdx);

  /**
     @brief Post-pass to update numerical splitting values from ranks.

     @param summaryFrame records the predictor types.
  */
  void splitUpdate(const class SummaryFrame* sf);
};

#endif
