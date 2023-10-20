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

#ifndef FOREST_FOREST_H
#define FOREST_FOREST_H

#include "dectree.h"
#include "leaf.h"
#include "typeparam.h"
#include "scoredesc.h"

#include <numeric>
#include <vector>
#include <complex>

/**
   @brief The decision forest as a read-only collection.
*/
class Forest {
  vector<DecTree> decTree; ///< New representation; ultimately constant.
  const ScoreDesc scoreDesc;
  const Leaf leaf;  //  const unique_ptr<class Leaf> leaf;
  const size_t noNode; ///< Inattainable node index.
  const unsigned int nTree;


  void dump(vector<vector<PredictorT>>& predTree,
            vector<vector<double>>& splitTree,
            vector<vector<size_t>>& lhDelTree,
	    vector<vector<double>>& scoreTree) const;
  
 public:

  static void init(PredictorT nPred) {
    DecNode::initMasks(nPred);
  }


  static void deInit() {
    DecNode::deInit();
    RankCount::unsetMasks();
  }


  /**
     @param decTree is built OTF.

     @param leaf_ may or may not be populated by caller.
   */
  Forest(vector<DecTree>&& decTree,
	 const tuple<double, double, string>& scoreDesc_,
	 Leaf&& leaf_);


  /**
     @brief Initializes walker state.  Ultimately deprecated.
   */
  void initWalkers(const class PredictFrame& trFrame);


  IndexT walkObs(const class PredictFrame& frame,
		 size_t obsIdx,
		 unsigned int tIdx) const {
    return decTree[tIdx].walkObs(frame, obsIdx);
  }

  
  /**
     @brief Maps leaf indices to the node at which they appear.
   */
  vector<IndexT> getLeafNodes(unsigned int tIdx,
			      IndexT extent) const;


  /**
     @brief Getter for 'nTree'.
     
     @return number of trees in the forest.
   */
  inline unsigned int getNTree() const {
    return nTree;
  }


  const vector<DecNode>& getNode(unsigned int tIdx) const {
    return decTree[tIdx].getNode();
  }

  
  size_t getNoNode() const {
    return noNode;
  }


  bool getLeafIdx(unsigned int tIdx,
		  IndexT nodeIdx,
		  IndexT& leafIdx) const {
    return decTree[tIdx].getLeafIdx(nodeIdx, leafIdx);
  }
  

  double getScore(unsigned int tIdx,
		  IndexT nodeIdx) const {
    return decTree[tIdx].getScore(nodeIdx);
  }


  //  const struct Leaf* getLeaf() const;
  const Leaf& getLeaf() const {
    return leaf;
  }

  
  /**
     @return vector of domininated leaf ranges, per node.
   */
  static vector<IndexRange> leafDominators(const vector<DecNode>& tree);


  /**
     @brief Computes a vector of leaf dominators for every tree.
   */  
  vector<vector<IndexRange>> leafDominators() const;


  /**
     @brief Computes an inattainable node index.

     @return maximum tree extent.
   */
  static size_t maxHeight(const vector<DecTree>& decTree);


  unique_ptr<ForestPredictionReg> makePredictionReg(const class Sampler* sampler,
						    const class Predict* predict,
						    bool reportAuxiliary = true);


  unique_ptr<ForestPredictionCtg> makePredictionCtg(const class Sampler* sampler,
						    const class Predict* predict,
						    bool reportAuxiliary = true);


  /**
     @brief Dumps forest-wide structure fields as per-tree vectors.
     
     Suitable for bridge-level diagnostic methods.

     @param[out] predTree outputs per-tree splitting predictors.

     @param[out] splitTree outputs per-tree splitting criteria.

     @param[out] lhDelTree outputs per-tree lh-delta values.

     @param[out] facSplitTree outputs per-tree factor encodings.
   */
  void dump(vector<vector<PredictorT> > &predTree,
            vector<vector<double> > &splitTree,
            vector<vector<size_t> > &lhDelTree,
	    vector<vector<double>>& scoreTree,
	    IndexT& dummy) const;
};


#endif
