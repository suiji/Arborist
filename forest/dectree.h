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

class PredictFrame;
class DecTree;


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


  /**
     @brief Unpacks according to front-end format.
   */
  static vector<DecTree> unpack(unsigned int nTree,
			 const double nodeExtent[],
			 const complex<double> nodes[],
			 const double score[],
			 const double facExtent[],
			 const unsigned char facSplit[],
			 const unsigned char facObserved[]);

  
  static vector<double> unpackDoubles(const double val[],
				      const size_t extent);


  static BV unpackBits(const unsigned char raw[],
		       size_t extent);

  
  static vector<DecNode> unpackNodes(const complex<double> nodes[],
				     size_t extent);


  const BV& getFacObserved() const {
    return facObserved;
  }


  const BV& getFacSplit() const {
    return facSplit;
  }


  IndexT walkObs(const PredictFrame* frame,
		 const bool trapUnobserved,
		 size_t obsIdx) const {
    IndexT idx = 0;
    IndexT delIdx = 0;
    do {
      delIdx = trapUnobserved ? decNode[idx].advanceTrap(frame, this, obsIdx) : decNode[idx].advance(frame, this, obsIdx);
      idx += delIdx;
    }  while (delIdx != 0);

    return idx;
  }


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
