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

#include "decnode.h"
#include "bv.h"
#include "typeparam.h"
#include "scoredesc.h"

#include <numeric>
#include <vector>
#include <complex>

/**
   @brief The decision forest as a read-only collection.
*/
class Forest {
  const unsigned int nTree; ///< # trees in chunk under training.
  const vector<vector<DecNode>> decNode;
  const vector<vector<double>> scores; //< Per node.
  const vector<unique_ptr<BV>> factorBits; ///< All factors known at training.
  const vector<unique_ptr<BV>> bitsObserved; ///< Factors observed at splitting.


  ScoreDesc scoreDesc; ///< Prediction only.


  void dump(vector<vector<PredictorT>>& predTree,
            vector<vector<double>>& splitTree,
            vector<vector<size_t>>& lhDelTree,
	    vector<vector<double>>& scoreTree) const;
  
 public:

  static void init(PredictorT nPred) {
    DecNode::init(nPred);
  }


  static void deInit() {
    DecNode::deInit();
  }

  
  /**
     Post-training constructor.
   */
  Forest(const vector<vector<DecNode>> decNode_,
	 vector<vector<double>> scores_,
	 vector<unique_ptr<BV>> factorBits_,
	 vector<unique_ptr<BV>> bitsObserved_,
	 const tuple<double, double, string>& scoreDesc_);


  /**
     @brief Maps leaf indices to the node at which they appear.
   */
  vector<IndexT> getLeafNodes(unsigned int tIdx,
			      IndexT extent) const;

  
  /**
     @brief Produces height vector from numeric representation.

     Front ends not supporting 64-bit integers can represent extent
     vectors as doubles.

     @return non-numeric height vector.
   */
  vector<size_t> produceHeight(const vector<size_t>& extent_) const;
  

  /**
     @brief Getter for 'nTree'.
     
     @return number of trees in the forest.
   */
  inline unsigned int getNTree() const {
    return nTree;
  }

  
  /**
     @brief Getter for node record vector.

     @return reference to node vector.
   */
  const vector<vector<DecNode>>& getNode() const {
    return decNode;
  }


  const vector<DecNode>& getNode(unsigned int tIdx) const {
    return decNode[tIdx];
  }


  /**
     @return vector of domininated leaf ranges, per node.
   */
  static vector<IndexRange> leafDominators(const vector<DecNode>& tree);


  /**
     @brief Computes a vector of leaf dominators for every tree.
   */  
  vector<vector<IndexRange>> leafDominators() const;


  inline const vector<unique_ptr<BV>>& getFactorBits() const {
    return factorBits;
  }

  
  inline const vector<unique_ptr<BV>>& getBitsObserved() const {
    return bitsObserved;
  }


  /**
     @brief Computes an inattainable node index.

     @return maximum tree extent.
   */
  size_t noNode() const;

  
  /**
     @return per-tree vector of scores.
   */
  const vector<vector<double>>& getTreeScores() const {
    return scores;
  }

  
  /**
     @brief Passes through to ScoreDesc method.
   */
  unique_ptr<class ForestScorer> makeScorer(const class ResponseReg* response,
					    const class Forest* forest,
					    const class Leaf* leaf,
					    const class PredictReg* predict,
					    vector<double> quantile) const;


  unique_ptr<class ForestScorer> makeScorer(const class ResponseCtg* response,
					    size_t nObs,
					    bool doProb) const;


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
