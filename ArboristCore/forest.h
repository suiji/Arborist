// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file forest.h

   @brief Data structures and methods for constructing and walking
       the decision trees.

   @author Mark Seligman
 */

#ifndef ARBORIST_FOREST_H
#define ARBORIST_FOREST_H

#include <vector>
#include <algorithm>

#include "decnode.h"

/**
   @brief To replace parallel array access.
 */
class TreeNode : public DecNode {
  static vector<double> splitQuant;

 public:

  /**
     @param[out] leafIdx outputs predictor index iff at leaf.
   */
  inline unsigned int advance(const double *rowT,
		       unsigned int &leafIdx) const {
    if (lhDel == 0) {
      leafIdx = predIdx;
      return 0;
    }
    else {
      return rowT[predIdx] <= splitVal.num ? lhDel : lhDel + 1;
    }
  }

  
  unsigned int advance(const class BVJagged *facSplit,
                       const unsigned int *rowT,
                       unsigned int tIdx,
                       unsigned int &leafIdx) const;
  

  unsigned int advance(const class FramePredict *framePredict,
                       const BVJagged *facSplit,
                       const unsigned int *rowFT,
                       const double *rowNT,
                       unsigned int tIdx,
                       unsigned int &leafIdx) const;



  static void Immutables(const vector<double> &feSplitQuant) {
    for (auto quant : feSplitQuant) {
      splitQuant.push_back(quant);
    }
  }

  
  static void DeImmutables() {
    splitQuant.clear();
  }
  
  

  /**
     @brief Post-pass to update numerical splitting values from ranks.

     @param frameTrain records the predictor types.

     @return void
  */
  void splitUpdate(const class FrameTrain *frameTrain,
                   const class BlockRanked *numRanked);

  /**
     @brief Initializer for tree node.
   */
  inline void init() {
    predIdx = lhDel = 0;
    splitVal.num = 0.0;
  }


  inline void setRank(const DecNode *decNode) {
    predIdx = decNode->predIdx;
    lhDel = decNode->lhDel;
    splitVal = decNode->splitVal;
    //    splitVal.rankRange = decNode->splitVal.rankRange;
  }


  /**
     @brief Copies decision node, converting offset to numeric value.

     @return void.
   */
  inline void setOffset(const DecNode *decNode) {
    predIdx = decNode->predIdx;
    lhDel = decNode->lhDel;
    splitVal = decNode->splitVal;
    //    splitVal.num = decNode->splitVal.offset;
  }


  /**
     @brief Initializes leaf node.

     @return void.
   */
  inline void setLeaf(unsigned int leafIdx) {
    predIdx = leafIdx;
    splitVal.num = 0.0;
  }


  /**
     @brief Indicates whether node is nonterminal.

     @return True iff lhDel value is nonzero.
   */
  inline bool Nonterminal() const {
    return lhDel != 0;
  }  


  inline unsigned int Pred() const {
    return predIdx;
  }


  inline unsigned int LHDel() const {
    return lhDel;
  }


  inline double Split() const {
    return splitVal.num;
  }

  
  /**
     @brief Multi-field accessor for a tree node.

     @param pred outputs the associated predictor index.

     @param lhDel outputs the increment to the LH descendant, if any.

     @param num outputs the numeric split value.

     @return void, with output reference parameters.
   */
  inline void RefNum(unsigned int &_pred,
                  unsigned int &_lhDel,
                  double &_num) const {
    _pred = predIdx;
    _lhDel = lhDel;
    _num = splitVal.num;
  }
};


/**
   @brief The decision forest as a read-only collection.
*/
class Forest {
  const unsigned int* nodeHeight;
  const unsigned int nTree;
  const TreeNode *treeNode;
  const unsigned int nodeCount;
  unique_ptr<class BVJagged> facSplit; // Consolidation of per-tree values.

  void dump(vector<vector<unsigned int> > &predTree,
            vector<vector<double> > &splitTree,
            vector<vector<unsigned int> > &lhDelTree) const;

  
 public:

  Forest(const unsigned int height_[],
         unsigned int _nTree,
         const TreeNode _treeNode[],
         unsigned int _facVec[],
         const unsigned int facHeight_[]);

  ~Forest();

  /**
     @brief Getter for 'nTree'.
     
     @return number of trees in the forest.
   */
  inline unsigned int getNTree() const {
    return nTree;
  }

  /**
     @brief Accessor for node records.

     @return pointer to base of node vector.
   */
  inline const TreeNode *Node() const {
    return treeNode;
  }

  
  /**
     @brief Accessor for split encodings.

     @return pointer to base of split-encoding vector.
   */
  inline const BVJagged *getFacSplit() const {
    return facSplit.get();
  }
  
  /**
     @brief Determines height of individual tree height.

     @param tIdx is the tree index.

     @return Height of tree.
   */
  inline size_t getNodeHeight(unsigned int tIdx) const {
    return nodeHeight[tIdx];
  }


  /**
     @brief Derives tree origins from the forest height vector
     and caches.

     @return vector of per-tree node starting offsets.
   */
  vector<size_t> cacheOrigin() const;

  /**
     @brief Dumps forest-wide structure fields as per-tree vectors.

     @param[out] predTree outputs per-tree splitting predictors.

     @param[out] splitTree outputs per-tree splitting criteria.

     @param[out] lhDelTree outputs per-tree lh-delta values.

     @param[out] facSplitTree outputs per-tree factor encodings.
   */
  void dump(vector<vector<unsigned int> > &predTree,
            vector<vector<double> > &splitTree,
            vector<vector<unsigned int> > &lhDelTree,
            vector<vector<unsigned int> > &facSplitTree) const;
};


/**
   @brief TreeNode block for crescent frame;
 */
class NBCresc {
  vector<TreeNode> treeNode;
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
     @brief Post-pass to update numerical splitting values from ranks.

     @param rowRank holds the presorted predictor values.
  */
  void splitUpdate(const class FrameTrain *frameTrain,
                   const class BlockRanked* numRanked);

  
  /**
     @brief Accessor for height vector.
   */
  const vector<size_t>& getHeight() const {
    return height;
  }


  /**
     @brief Sets looked-up nonterminal node to values passed.

     @return void.
  */
  inline void branchProduce(unsigned int nodeIdx,
                            const DecNode *decNode,
                            bool isFactor) {
    if (isFactor)
      treeNode[treeFloor + nodeIdx].setOffset(decNode);
    else
      treeNode[treeFloor + nodeIdx].setRank(decNode);
  }

  /**
    @brief Sets looked-up leaf node to leaf index passed.

    @return void.

  */
  inline void leafProduce(unsigned int nodeIdx,
                          unsigned int leafIdx) {
    treeNode[treeFloor + nodeIdx].setLeaf(leafIdx);
  }
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
                  unsigned int bitEnd,
                  unsigned int tIdx);

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
  unique_ptr<NBCresc> nbCresc;
  unique_ptr<FBCresc> fbCresc;

 public:
  /**
     @brief Constructs block of trees for crescent forest, wrapping
     vectors allocated by the front-end bridge.

     @param[in] treeChunk is the number of trees to train.
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
                  unsigned int bitEnd,
                  unsigned int tIdx);

  /**
     @brief Allocates and initializes sufficient nodes for current tree.

     @param tIdx is the block-relative tree index.

     @param nodeCount is the number of nodes.
   */
  void treeInit(unsigned int tIdx,
                unsigned int nodeCount);

  void splitUpdate(const class FrameTrain *frameTrain,
                   const class BlockRanked *numRanked);

  void nonTerminal(const class FrameTrain *frameTrain,
                   unsigned int idx,
                   const DecNode *decNode);

  /**
     @brief Outputs raw byes of node vector.

     @return void.
   */
  void cacheNodeRaw(unsigned char rawOut[]) const {
    nbCresc->dumpRaw(rawOut);
  }

  /**
     @brief Outputs raw byes of node vector.

     @return void.
   */
  void cacheFacRaw(unsigned char rawOut[]) const {
    fbCresc->dumpRaw(rawOut);
  }


  /**
   */
  const vector<size_t>& getNodeHeight() const {
    return nbCresc->getHeight();
  }
  

  /**
     @brief Exposes fac-origin vector.
   */
  const vector<size_t>& getFacHeight() const {
    return fbCresc->getHeight();
  }

  /**
    @brief Sets tree node as terminal.

    @param nodeIdx is a tree-relative node index.

    @param leafIdx is a tree-relative leaf index.

    @return void.

  */
  void terminal(unsigned int nodeIdx,
                unsigned int leafIdx);
};

#endif
