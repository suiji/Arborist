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


 CartNode(complex<double> pair) :
  TreeNode(pair) {
  }
  
  /**
     @brief Resets nonterminal with specified index delta.
   */
  //  inline void setDelIdx(IndexT delIdx) {
  //this->delIdx = delIdx;
  //}


  /**
     @return pretree index of true branch target.
   */
  inline IndexT getIdTrue(IndexT ptId) const {
    return isNonterminal() ? ptId + getDelIdx() : 0;
  }
  

  /**
     @return pretree index of false branch target.
   */
  inline IndexT getIdFalse(IndexT ptId) const {
    return isNonterminal() ? ptId + getDelIdx() + 1 : 0;
  }


  /**
     @brief Advancers pass through to the base class.
   */
  inline IndexT advance(const double* rowT) const {
    return TreeNode::advance(rowT);
  }


  inline IndexT advance(const vector<unique_ptr<class BV>>& factorBits,
			const IndexT* rowT,
			unsigned int tIdx) const {
    return TreeNode::advance(factorBits, rowT, tIdx);
  }


  inline IndexT advance(const class Predict* predict,
			const vector<unique_ptr<class BV>>& factorBits,
			const IndexT* rowFT,
			const double *rowNT,
			unsigned int tIdx) const {
    return TreeNode::advance(predict, factorBits, rowFT, rowNT, tIdx);
  }
};

#endif
