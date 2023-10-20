// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file dectree.h

   @brief Decision tree representation.

   @author Mark Seligman
 */


#ifndef FOREST_DECTREE_H
#define FOREST_DECTREE_H

#include "decnode.h"
#include "bv.h"
#include "typeparam.h"

class DecTree {
  const vector<DecNode> decNode; ///< Decision nodes.
  const BV facSplit; ///< Categories splitting node.
  const BV facObserved; ///< Categories observed at node.
  const vector<double> nodeScore; ///< Per-node score.

public:

  DecTree(const vector<DecNode>& decNode_,
	  const BV& facSplit_,
	  const BV& facObserved_,
	  const vector<double>& nodeScore_);

  ~DecTree();

  
  IndexT (DecTree::* obsWalker)(const class PredictFrame&, size_t) const;
  

  void setObsWalker(PredictorT nPredNum);

  
  IndexT walkObs(const class PredictFrame& frame,
		 size_t obsIdx) const {
    return (this->*DecTree::obsWalker)(frame, obsIdx);
  }

  
  IndexT obsNum(const class PredictFrame& frame,
		size_t obsIdx) const;


  IndexT obsFac(const class PredictFrame& frame,
		size_t obsIdx) const;


  IndexT obsMixed(const class PredictFrame& frame,
		  size_t obsIdx) const;


  size_t nodeCount() const {
    return decNode.size();
  }


  const vector<DecNode>& getNode() const {
    return decNode;
  }


  bool getLeafIdx(IndexT nodeIdx,
		  IndexT& leafIdx) const {
    return decNode[nodeIdx].getLeafIdx(leafIdx);
  }
  
  
  double getScore(IndexT nodeIdx) const {
    return nodeScore[nodeIdx];
  }

  
  double getSplitNum(IndexT nodeIdx) const {
    return decNode[nodeIdx].getSplitNum();
  }


  IndexT getDelIdx(IndexT nodeIdx) const {
    return decNode[nodeIdx].getDelIdx();
  }


  IndexT getPredIdx(IndexT nodeIdx) const {
    return decNode[nodeIdx].getPredIdx();
  }
};

#endif
