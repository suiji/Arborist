// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file cartnode.h

   @brief Data structures and methods implementing CART tree nodes.

   @author Mark Seligman
 */


#ifndef CART_CARTNODE_H
#define CART_CARTNODE_H

#include "treenode.h"


/**
   @brief To replace parallel array access.
 */
struct CartNode : public TreeNode {

  /**
     @brief Nodes must be explicitly set to non-terminal (delIdx != 0).
   */
 CartNode() : TreeNode() {
  }
  
  /**
     @brief Resets nonterminal with specified index delta.
   */
  inline void setDelIdx(IndexT delIdx) {
    this->delIdx = delIdx;
  }


  /**
     @return pretree index of true branch target.
   */
  inline IndexT getIdTrue(IndexT ptId) const {
    return isNonterminal() ? ptId + delIdx : 0;
  }
  

  /**
     @return pretree index of false branch target.
   */
  inline IndexT getIdFalse(IndexT ptId) const {
    return isNonterminal() ? ptId + delIdx + 1 : 0;
  }


  /**
     @brief Advances to next node when observations are all numerical.

     @param rowT is a row base within the transposed numerical set.

     @param[out] leafIdx outputs predictor index iff at terminal.

     @return delta to next node, if nonterminal, else zero.
   */
  inline IndexT advance(const double *rowT,
			IndexT& leafIdx) const {
    auto predIdx = getPredIdx();
    if (delIdx == 0) {
      leafIdx = predIdx;
      return 0;
    }
    else {
      return rowT[predIdx] <= getSplitNum() ? delIdx : delIdx + 1;
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
  IndexT advance(const class BVJagged *facSplit,
		 const IndexT* rowT,
		 unsigned int tIdx,
		 IndexT& leafIdx) const;
  
  /**
     @brief Node advancer, as above, but for mixed observation.

     Parameters as above, along with:

     @param rowNT contains the transponsed numerical observations.

     @return terminal/nonterminal : 0 / delta to next node.
   */
  IndexT advance(const class PredictFrame* blockFrame,
		 const BVJagged *facSplit,
		 const IndexT* rowFT,
		 const double *rowNT,
		 unsigned int tIdx,
		 IndexT& leafIdx) const;
};

#endif
