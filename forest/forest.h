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

#include "prediction.h"
#include "dectree.h"
#include "leaf.h"
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
  static const size_t scoreChunk; ///< Score block dimension.
  static const unsigned int seqChunk;  ///< Effort to minimize false sharing.

  vector<DecTree> decTree; ///< New representation; ultimately constant.
  const ScoreDesc scoreDesc;
  const Leaf leaf;  //  const unique_ptr<class Leaf> leaf;
  const size_t noNode; ///< Inattainable node index.
  const unsigned int nTree;

  // Prediction state:
  size_t nObs;
  PredictorT nPredFac;
  PredictorT nPredNum;
  vector<IndexT> idxFinal; ///< Final walk index; typically terminal.
  size_t blockStart; ///< Stripmine bound.
  vector<CtgT> trFac; ///< OTF transposed factor observations.
  vector<double> trNum; ///< OTF transposed numeric observations.

  
  /**
     @brief Initializes prediction state from the predictor frame.
   */
  void initPrediction(const class RLEFrame* rleFrame);


  void setFinalIdx(size_t obsIdx, unsigned int tIdx, IndexT finalIdx) {
    idxFinal[nTree * (obsIdx - blockStart) + tIdx] = finalIdx;
  }


  void predict(const class Sampler* sampler,
	       const class RLEFrame* rleFrame,
	       ForestPrediction* prediction);


  size_t predictBlock(const class Sampler* sampler,
		      const RLEFrame* rleFrame,
		      ForestPrediction* prediction,
		      size_t rowStart,
		      size_t rowEnd,
		      vector<size_t>& trIdx);


  /**
     @brief Predicts a block of observations in parallel.
   */
  void predictObs(const class Sampler* sampler,
		  ForestPrediction* prediction,
		  size_t obsStart,
		  size_t span);

  
  /**
     @brief Predicts along each tree.

     @param rowStart is the absolute starting row for the block.
  */
  void walkTree(const class Sampler* sampler,
		size_t obsStart,
		size_t obsEnd);


  /**
     @brief Populates blocks of predictor values in walkable order.

     @param[in,out] idxTr is the most-recently accessed RLE index, by predictor.

     @param rowStart is the starting source row.

     @param rowExtent is the total number of rows to transpose.
   */
  void transpose(const class RLEFrame* rleFrame,
		 vector<size_t>& idxTr,
		 size_t rowStart,
		 size_t rowExtent);


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


  void setBlockStart(size_t blockStart);


  /**
     @brief Computes block-relative position for a predictor.

     @param[out] thisIsFactor outputs true iff predictor is factor-valued.

     @return block-relative index.
   */
  inline unsigned int getIdx(PredictorT predIdx, bool& predIsFactor) const {
    predIsFactor = isFactor(predIdx);
    return predIsFactor ? predIdx - nPredNum : predIdx;
  }

  
  inline bool isFactor(PredictorT predIdx) const {
    return predIdx >= nPredNum;
  }

  
  /**
     @brief Computes pointer to base of transposed numeric row.

     @param row is the row number.

     @return base address for numeric values at observation.
  */
  inline const double* baseNum(size_t row) const {
    return &trNum[(row - blockStart) * nPredNum];
  }


  /**
     @brief As above, but row of factor varlues.

     @return base address for factor values at observation.
   */
  inline const CtgT* baseFac(size_t row) const {
    return &trFac[(row - blockStart) * nPredFac];
  }


  bool getFinalIdx(size_t obsIdx, unsigned int tIdx, IndexT& nodeIdx) const;


  bool isLeafIdx(size_t row,
		 unsigned int tIdx,
		 IndexT& leafIdx) const;

  
  bool isNodeIdx(size_t obsIdx,
		 unsigned int tIdx,
		 double& score) const;


  IndexT walkObs(size_t obsIdx,
		 unsigned int tIdx) {
    return decTree[tIdx].walkObs(this, obsIdx);
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


  unique_ptr<ForestPredictionReg> predictReg(const class Sampler* sampler,
					     const class RLEFrame* rleFrame);


  unique_ptr<ForestPredictionCtg> predictCtg(const class Sampler* sampler,
					     const class RLEFrame* rleFrame);


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
