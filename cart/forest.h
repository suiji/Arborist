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

#ifndef CORE_FOREST_H
#define CORE_FOREST_H

#include <vector>
#include <algorithm>

#include "decnode.h"

/**
   @brief To replace parallel array access.
 */
class TreeNode : public DecNode {
  static vector<double> splitQuant; // Where within CDF to split.

 public:

  /**
     @brief Getter for splitting predictor.

     @return splitting predictor index.
   */
  inline unsigned int getPredIdx() const {
    return criterion.predIdx;
  }


  /**
     @brief Getter for numeric splitting value.

     @return splitting value.
   */
  inline auto getSplitNum() const {
    return criterion.getNumVal();
  }


  /**
     @return first bit position of split.
   */
  inline auto getSplitBit() const {
    return criterion.getBitOffset();
  }
  

  /**
     @brief Advances to next node when observations are all numerical.

     @param rowT is a row base within the transposed numerical set.

     @param[out] leafIdx outputs predictor index iff at terminal.

     @return delta to next node, if nonterminal, else zero.
   */
  inline unsigned int advance(const double *rowT,
                              unsigned int &leafIdx) const {
    auto predIdx = getPredIdx();
    if (lhDel == 0) {
      leafIdx = predIdx;
      return 0;
    }
    else {
      return rowT[predIdx] <= getSplitNum() ? lhDel : lhDel + 1;
    }
  }

  /**
     @brief Node advancer, as above, but for all-categorical observations.

     @param facSplit accesses the per-tree packed factor-splitting vectors.

     @param rowT holds the transposed factor-valued observations.

     @param tIdx is the tree index.

     @param leafIdx as above.

     @return terminal/nonterminal : 0 / delta to next node.
   */
  unsigned int advance(const class BVJagged *facSplit,
                       const unsigned int *rowT,
                       unsigned int tIdx,
                       unsigned int &leafIdx) const;
  
  /**
     @brief Node advancer, as above, but for mixed observation.

     Parameters as above, along with:

     @param rowNT contains the transponsed numerical observations.

     @return terminal/nonterminal : 0 / delta to next node.
   */
  unsigned int advance(const class PredictFrame* blockFrame,
                       const BVJagged *facSplit,
                       const unsigned int *rowFT,
                       const double *rowNT,
                       unsigned int tIdx,
                       unsigned int &leafIdx) const;



  /**
     @brief Builds static quantile splitting vector from front-end specification.

     @param feSplitQuant specifies the splitting quantiles for numerical predictors.
   */
  static void Immutables(const vector<double> &feSplitQuant) {
    for (auto quant : feSplitQuant) {
      splitQuant.push_back(quant);
    }
  }

  /**
     @brief Empties the static quantile splitting vector.
   */
  static void DeImmutables() {
    splitQuant.clear();
  }
  
  
  /**
     @brief Derives split values for a numerical predictor by synthesizing
     a fractional intermediate rank and interpolating.

     Paramters as with low-level implementation.
   */
  void setQuantRank(const class SummaryFrame* sf);

  /**
     @brief Copies decision node, converting offset to numeric value.

     @param decNode encodes the splitting specification.
   */
  inline void setBranch(IndexType lhDel,
                        const SplitCrit& crit) {
    this->lhDel = lhDel;
    criterion = crit;
  }


  /**
     @brief Initializes leaf node.

     @param leafIdx is the tree-relative leaf index.
   */
  inline void setLeaf(unsigned int leafIdx) {
    criterion.predIdx = leafIdx;
    criterion.val.num = 0.0;
  }


  /**
     @brief Indicates whether node is nonterminal.

     @return True iff lhDel value is nonzero.
   */
  inline bool Nonterminal() const {
    return lhDel != 0;
  }  


  /**
     @brief Getter for lh-delta value.

     @return lhDel value.
   */
  inline auto getLHDel() const {
    return lhDel;
  }


  /**
     @brief Multi-field accessor for a tree node.

     @param pred outputs the associated predictor index.

     @param lhDel outputs the increment to the LH descendant, if any.

     @param num outputs the numeric split value.

     @return void, with output reference parameters.
   */
  inline void RefNum(unsigned int &pred,
                     IndexType &lhDel,
                     double &num) const {
    pred = getPredIdx();
    lhDel = this->lhDel;
    num = getSplitNum();
  }
};


/**
   @brief The decision forest as a read-only collection.
*/
class Forest {
  const unsigned int* nodeHeight;
  const unsigned int nTree;
  const TreeNode *treeNode;
  unique_ptr<class BVJagged> facSplit; // Consolidation of per-tree values.

  void dump(vector<vector<unsigned int> > &predTree,
            vector<vector<double> > &splitTree,
            vector<vector<IndexType> > &lhDelTree) const;

  
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
     @brief Getter for node records.

     @return pointer to base of node vector.
   */
  inline const TreeNode *getNode() const {
    return treeNode;
  }

  
  /**
     @brief Accessor for split encodings.

     @return pointer to base of split-encoding vector.
   */
  inline const BVJagged* getFacSplit() const {
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
     
     Suitable for bridge-level diagnostic methods.

     @param[out] predTree outputs per-tree splitting predictors.

     @param[out] splitTree outputs per-tree splitting criteria.

     @param[out] lhDelTree outputs per-tree lh-delta values.

     @param[out] facSplitTree outputs per-tree factor encodings.
   */
  void dump(vector<vector<unsigned int> > &predTree,
            vector<vector<double> > &splitTree,
            vector<vector<IndexType> > &lhDelTree,
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
     @brief Computes unit size for cross-compatibility of serialization.
   */
  static constexpr size_t nodeSize() {
    return sizeof(TreeNode);
  }
  

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
  inline void branchProduce(unsigned int nodeIdx,
                            IndexType lhDel,
                            const SplitCrit& crit) {
    treeNode[treeFloor + nodeIdx].setBranch(lhDel, crit);
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

  /**
     @brief Precipitates production of a branch node in the crescent forest.

     @param frame summarizes the training observations.

     @param idx is a tree-relative node index.

     @parm decNode contains the value to set.
   */
  void nonTerminal(IndexType idx,
                   IndexType lhDel,
                   const SplitCrit& crit);

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
