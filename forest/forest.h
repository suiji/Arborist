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
  vector<size_t> extents; // # nodes in each tree.
  size_t treeFloor; // Block-relative index of current tree floor.

public:

  void appendExtent(IndexT extent) {
    extents.push_back(extent);
  }


  void consumeNodes(const vector<DecNode>& nodes,
		    IndexT height) {
    copy(nodes.begin(), nodes.begin() + height, back_inserter(treeNode));
    appendExtent(height);
  }

  
  size_t getNodeBytes() const {
    return treeNode.size() * sizeof(DecNode);
  }


  const vector<size_t>& getExtents() const {
    return extents;
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
};


/**
   @brief Manages the crescent factor blocks.
 */
class FBCresc {
  vector<unsigned int> fac;  // Factor-encoding bit vector.
  vector<size_t> extents; // Extent of bit encoding, per tree.
  
public:
  
  /**
     @brief Consumes factor bit vector and notes height.

     @param splitBits is the bit vector.

     @param bitEnd is the final bit position referenced.
   */
  void appendBits(const class BV& splitBits,
		  size_t bitEnd);

  
  const vector<size_t>& getExtents() const {
    return extents;
  }
  

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
  const vector<size_t> nodeExtent; // Per-tree size of node encoding. 
  const DecNode* treeNode; // Post-training only.
  const double* scores; // " "

  unique_ptr<NodeCresc> nodeCresc; // Crescent node block:  training only.
  unique_ptr<FBCresc> fbCresc; // Crescent factor-summary block.
  vector<double> scoresCresc;

  unique_ptr<class BVJagged> facSplit; // Consolidation of per-tree values.


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
    nodeCresc(make_unique<NodeCresc>()),
    fbCresc(make_unique<FBCresc>()) {
  }


  /**
     Post-training constructor.
   */
  Forest(unsigned int nTree_,
	 const double nodeExtent_[],
	 const DecNode treeNode_[],
	 const double* scores_,
	 const double facExtent_[],
         unsigned int facVec[]);


  size_t getNodeBytes() const {
    return nodeCresc->getNodeBytes();
  };


  const vector<size_t>& getFacExtents() const {
    return fbCresc->getExtents();
  }
  
  
  const vector<size_t>& getNodeExtents() const {
    return nodeCresc->getExtents();
  }

  /**
     @brief Produces extent vector from numeric representation.

     Front ends not supporting 64-bit integers can represent extent
     vectors as doubles.

     @return non-numeric extent vector.
   */


  vector<size_t> produceExtent(const double extent_[]) const;


  /**
     @brief Produces height vector from numeric representation.

     Front ends not supporting 64-bit integers can represent extent
     vectors as doubles.

     @return non-numeric height vector.
   */
  vector<size_t> produceHeight(const double extent_[]) const;
  

  const vector<double>& getScores() const {
    return scoresCresc;
  }
  
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


  size_t getScoreSize() const {
    return scoresCresc.size();
  }
  
  
  void cacheScore(double scoreOut[]) {
    for (size_t i = 0; i < scoresCresc.size(); i++)
      scoreOut[i] = scoresCresc[i];
  }
  

  /**
     @brief Outputs raw byes of node vector.
   */
  void cacheNodeRaw(unsigned char rawOut[]) const {
    nodeCresc->dumpRaw(rawOut);
  }


  void consumeTree(const vector<DecNode>& nodes,
		   const vector<double>& scores_,
		   IndexT height) {
    nodeCresc->consumeNodes(nodes, height);
    copy(scores_.begin(), scores_.begin() + height, back_inserter(scoresCresc));
  }


  /**
     @brief Wrapper for bit vector appending.

     @param splitBits encodes bits maintained for the current tree.

     @param bitEnd is the final referenced bit position.
   */
  void consumeBits(const class BV& splitBits,
		   size_t bitEnd) {
    fbCresc->appendBits(splitBits, bitEnd);
  }


  /**
     @brief Post-pass to update numerical splitting values from ranks.

     @param summaryFrame records the predictor types.
  */
  void splitUpdate(const class TrainFrame* trainFrame) {
    nodeCresc->splitUpdate(trainFrame);
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
     @return maximum tree extent.
   */
  size_t maxTreeHeight() const;
  

  /**
     @brief Derives tree origins from the forest height vector
     and caches.

     @return vector of per-tree node starting offsets.
   */
   vector<size_t> treeOrigins() const;

  
  /**
     @return per-tree vector of scores.
   */
  vector<vector<double>> produceScores() const;


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
