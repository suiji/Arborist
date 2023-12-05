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


struct CartNode;
class DecTree;
class PredictFrame;

// EXIT
typedef IndexT (CartNode::*BrancherT)(const DecTree*, const PredictFrame*, size_t) const;

/**
   @brief To replace parallel array access.
 */
struct CartNode : public TreeNode {

  BrancherT brancher; // EXIT

  /**
     @brief Nodes must be explicitly set to non-terminal (delIdx != 0).
   */
 CartNode() : TreeNode() {
  }


  CartNode(complex<double> pair) :
    TreeNode(pair) {
  }

  /**
     @return pretree index of true branch target.
   */
  IndexT getIdTrue(IndexT ptId) const {
    return isNonterminal() ? ptId + getDelIdx() : 0;
  }
  

  /**
     @return pretree index of false branch target.
   */
  IndexT getIdFalse(IndexT ptId) const {
    return isNonterminal() ? ptId + getDelIdx() + 1 : 0;
  }


  IndexT advance(const PredictFrame* frame,
		 const DecTree* decTree,
		 size_t obsIdx) const;
};

#endif
