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

#include <vector>

/**
   @brief struct CartNode block for crescent frame;
 */
class NodeCresc {
  vector<DecNode> treeNode;
  size_t treeFloor; // Block-relative index of current tree floor.

public:

  /**
     @brief Allocates nodes for current tree.  Pre-initializes for random access.

     @param tIdx is the block-relative tree index.

     @param nodeCount is the number of tree nodes.
   */
  void treeInit(IndexT nodeCount) {
    treeFloor = treeNode.size();
    DecNode tn;
    treeNode.insert(treeNode.end(), nodeCount, tn);
  }

  
  /**
     @brief Writes the factor value length as a dummy leaf appended to the tree.
   */
  void treeFinish(size_t facEnd) {
    DecNode tn;
    tn.setLeaf(facEnd);
    treeNode.insert(treeNode.end(), 1, tn);
  }


  size_t getNodeBytes() const {
    return treeNode.size() * sizeof(DecNode);
  }
  

  /**
     @brief Copies treeNode contents by byte.

     @param[out] nodeRaw outputs the raw contents.
  */
  void dumpRaw(unsigned char nodeRaw[]) const {
    unsigned char* nodeBase = (unsigned char*) &treeNode[0];
    for (size_t i = 0; i < treeNode.size() * sizeof(DecNode); i++) {
      nodeRaw[i] = nodeBase[i];
    }
  }


  /**
     @brief Tree-level dispatch to low-level member.

     Parameters as with low-level implementation.
  */
  void splitUpdate(const class TrainFrame* trainFrame) {
    for (auto & tn : treeNode) {
      tn.setQuantRank(trainFrame);
    }
  }


  /**
     @brief Sets looked-up node to values passed.

     @param nodeIdx is a tree-relative node index.

     @param decNode contains the value to set.
  */
  void produce(IndexT nodeIdx,
	       const DecNode& decNode) {
    treeNode[treeFloor + nodeIdx] = decNode;
  }


  /**
    @brief Sets looked-up leaf node to leaf index passed.
  */
  void setScore(IndexT nodeIdx,
		double score) {
    treeNode[treeFloor + nodeIdx].setScore(score);
  }
};


/**
   @brief Manages the crescent factor blocks.
 */
class FBCresc {
  vector<unsigned int> fac;  // Factor-encoding bit vector.

public:
  
  /**
     @brief Consumes factor bit vector and notes height.

     @param splitBits is the bit vector.

     @param bitEnd is the final bit position referenced.

     @return number of native bit-vector slots subsumed.
   */
  size_t appendBits(const class BV& splitBits,
		    size_t bitEnd);


  size_t getFactorBytes() const {
    return fac.size() * sizeof(unsigned int);
  }
  

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
  void dumpRaw(unsigned char facRaw[]) const {
    for (size_t i = 0; i < fac.size() * sizeof(unsigned int); i++) {
      facRaw[i] = ((unsigned char*) &fac[0])[i];
    }
  }
};


/**
   @brief The decision forest as a read-only collection.
*/
class Forest {
  const unsigned int nTree;
  const DecNode* treeNode; // Post-training only.
  vector<vector<size_t>> leafNode; // Per-tree vector of leaf indices.

  unique_ptr<NodeCresc> nodeCresc; // Crescent node block:  training only.
  unique_ptr<FBCresc> fbCresc; // Crescent factor-summary block.
  vector<IndexT> leafCresc; // Tree-relative indices of leaves.

  unique_ptr<class BVJagged> facSplit; // Consolidation of per-tree values.
  vector<size_t> treeHeight;

  
  /**
     @brief Collects leaf nodes across forest, by tree.
   */
  vector<vector<size_t>> leafForest() const;


  /**
     @brief builds a forest-wide map of factor heights.

     Eliminates temporary pseudo-leaves during construction.

     @return jagged representation of bit map.
   */
  unique_ptr<BVJagged> splitFactors(unsigned int facVe[]);

  
  /**
     @brief Builds a vector of tree heights.
   */
  vector<size_t> treeHeights() const;


  void dump(vector<vector<PredictorT> > &predTree,
            vector<vector<double> > &splitTree,
            vector<vector<IndexT> > &lhDelTree) const;

  
 public:

  /**
     @brief Training constructor.
   */
  Forest(unsigned int nTree_) :
    nTree(nTree_),
    treeNode(nullptr),
    leafNode(vector<vector<size_t>>(0)),
    nodeCresc(make_unique<NodeCresc>()),
    fbCresc(make_unique<FBCresc>()) {
  }


  /**
     Post-training constructor.
   */
  Forest(unsigned int nTree_,
	 const DecNode _treeNode[],
         unsigned int facVec[]);


  size_t getNodeBytes() const {
    return nodeCresc->getNodeBytes();
  };
  
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
  inline const DecNode* getNode() const {
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
     @brief Allocates and initializes sufficient nodes for current tree.

     @param tIdx is the block-relative tree index.

     @param nodeCount is the number of nodes.
   */
  void treeInit(IndexT nodeCount) {
    nodeCresc->treeInit(nodeCount);
  }


  /**
     @brief Outputs raw byes of node vector.
   */
  void cacheNodeRaw(unsigned char rawOut[]) const {
    nodeCresc->dumpRaw(rawOut);
  }


  /**
     @brief Precipitates production of a branch node in the crescent forest.

     @param nodeIdx is a tree-relative node index.

     @parm decNode contains the value to set.
  */
  void nodeProduce(IndexT nodeIdx,
		   const DecNode& decNode) {
    nodeCresc->produce(nodeIdx, decNode);
  }


  /**
     @brief Adds the index of a leaf node.
   */
  void setTerminal(IndexT nodeIdx) {
    leafCresc.push_back(nodeIdx);
  }


  void setScores(const vector<double>& score) {
    IndexT leafIdx = 0;
    for (IndexT nodeIdx : leafCresc) {
      nodeCresc->setScore(nodeIdx, score[leafIdx++]);
    }
    leafCresc.clear();
  }

  
  /**
     @brief Post-pass to update numerical splitting values from ranks.

     @param summaryFrame records the predictor types.
  */
  void splitUpdate(const class TrainFrame* trainFrame) {
    nodeCresc->splitUpdate(trainFrame);
  }


  
  /**
     @brief Wrapper for bit vector appending.

     @param splitBits encodes bits maintained for the current tree.

     @param bitEnd is the final referenced bit position.
   */
  void appendBits(const class BV& splitBits,
                  size_t bitEnd) {
    size_t nSlot = fbCresc->appendBits(splitBits, bitEnd);
    nodeCresc->treeFinish(nSlot);
  }

  
  size_t getFactorBytes() const {
    return fbCresc->getFactorBytes();
  }


  /**
     @brief Dumps raw splitting values for factors.

     @param[out] rawOut outputs the raw bytes of factor split values.
   */
  void cacheFacRaw(unsigned char rawOut[]) const {
    fbCresc->dumpRaw(rawOut);
  }


  /**
     @brief Derives tree origins from the forest height vector
     and caches.

     @return vector of per-tree node starting offsets.
   */
  vector<size_t> treeOrigins() const;

  
  /**
     @return per-tree vector of leaf scores.
   */
  vector<vector<double>> getScores() const;
  

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
            vector<vector<IndexT> > &lhDelTree,
            vector<vector<PredictorT> > &facSplitTree) const;
};


#endif
