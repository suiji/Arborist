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
   @brief struct CartNode block for crescent frame;
 */
class NodeCresc {
  vector<DecNode> treeNode;
  vector<size_t> extents; // # nodes in each tree.

public:


  void consumeNodes(const vector<DecNode>& nodes,
		    IndexT height) {
    copy(nodes.begin(), nodes.begin() + height, back_inserter(treeNode));
    extents.push_back(height);
  }


  const vector<size_t>& getExtents() const {
    return extents;
  }
  

  void dump(complex<double> nodeComplex[]) const {
    for (size_t i = 0; i < treeNode.size(); i++) {
      treeNode[i].dump(nodeComplex[i]);
    }
  }


  /**
     @brief Tree-level dispatch to low-level member.

     Parameters as with low-level implementation.
  */
  void splitUpdate(const class PredictorFrame* frame) {
    for (auto & tn : treeNode) {
      tn.setQuantRank(frame);
    }
  }
};


/**
   @brief Manages the crescent factor blocks.
 */
class FBCresc {
  vector<BVSlotT> splitBits;  // Agglomerates per-tree factor bit vectors.
  vector<BVSlotT> observedBits;
  vector<size_t> extents; // Per-tree extent of bit encoding in BVSlotT units.
  
public:
  
  /**
     @brief Consumes factor bit vector and notes height.

     @param splitBits is the bit vector.

     @param bitEnd is the final referenced bit position.
   */
  void appendBits(const class BV& splitBits_,
		  const class BV& observedBits_,
		  size_t bitEnd);

  
  const vector<size_t>& getExtents() const {
    return extents;
  }
  

  size_t getFactorBytes() const {
    return splitBits.size() * sizeof(BVSlotT);
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
  void dumpSplitBits(unsigned char facRaw[]) const {
    if (splitBits.empty())
      return;
    const unsigned char* bvRaw = reinterpret_cast<const unsigned char*>(&splitBits[0]);
    for (size_t i = 0; i < splitBits.size() * sizeof(BVSlotT); i++) {
      facRaw[i] = bvRaw[i];
    }
  }


  /**
     @brief Dumps factor bits as raw data.

     @param[out] facRaw outputs the raw factor data.
   */
  void dumpObserved(unsigned char observedRaw[]) const {
    if (observedBits.empty())
      return;
    const unsigned char* bvRaw = reinterpret_cast<const unsigned char*>(&observedBits[0]);
    for (size_t i = 0; i < observedBits.size() * sizeof(BVSlotT); i++) {
      observedRaw[i] = bvRaw[i];
    }
  }
};


/**
   @brief The decision forest as a read-only collection.
*/
class Forest {
  const unsigned int nTree; ///< # trees in chunk under training.
  const vector<vector<DecNode>> decNode;
  const vector<vector<double>> scores; //< Per node.
  const vector<unique_ptr<BV>> factorBits; ///< All factors known at training.
  const vector<unique_ptr<BV>> bitsObserved; ///< Factors observed at splitting.

  // Crescent data structures:  training only.
  unique_ptr<NodeCresc> nodeCresc; ///< Crescent node block.
  unique_ptr<FBCresc> fbCresc; ///< Crescent factor-summary block.
  vector<double> scoresCresc; ///< Crescent score block.

  ScoreDesc scoreDesc; ///< Prediction only.


  void dump(vector<vector<PredictorT>>& predTree,
            vector<vector<double>>& splitTree,
            vector<vector<size_t>>& lhDelTree,
	    vector<vector<double>>& scoreTree) const;
  
 public:

  /**
     @brief Training constructor.
   */
  Forest(unsigned int nTree_) :
    nTree(nTree_),
    nodeCresc(make_unique<NodeCresc>()),
    fbCresc(make_unique<FBCresc>()) {
  }


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


  const vector<size_t>& getFacExtents() const {
    return fbCresc->getExtents();
  }
  

  const vector<size_t>& getNodeExtents() const {
    return nodeCresc->getExtents();
  }


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
     @brief Obtains node count from score vector.
   */
  size_t getNodeCount() const {
    return scoresCresc.size();
  }
  
  
  void cacheScore(double scoreOut[]) {
    for (size_t i = 0; i < scoresCresc.size(); i++)
      scoreOut[i] = scoresCresc[i];
  }
  

  void cacheNode(complex<double> complexOut[]) const {
    nodeCresc->dump(complexOut);
  }


  void consumeTree(const vector<DecNode>& nodes,
		   const vector<double>& scores_) {
    IndexT height = nodes.size();
    nodeCresc->consumeNodes(nodes, height);
    copy(scores_.begin(), scores_.begin() + height, back_inserter(scoresCresc));
  }


  /**
     @brief Wrapper for bit vector appending.

     @param splitBits encodes bits maintained for the current tree.

     @param bitEnd is the final referenced bit position.
   */
  void consumeBits(const class BV& splitBits,
		   const class BV& observedBits,
		   size_t bitEnd) {
    fbCresc->appendBits(splitBits, observedBits, bitEnd);
  }


  /**
     @brief Post-pass to update numerical splitting values from ranks.

     @param summaryFrame records the predictor types.
  */
  void splitUpdate(const class PredictorFrame* frame) {
    nodeCresc->splitUpdate(frame);
  }

  
  size_t getFactorBytes() const {
    return fbCresc->getFactorBytes();
  }


  /**
     @brief Dumps raw splitting values for factors.

     @param[out] rawOut outputs the raw bytes of factor split values.
   */
  void cacheFacRaw(unsigned char rawOut[]) const {
    fbCresc->dumpSplitBits(rawOut);
  }


  void cacheObservedRaw(unsigned char observedOut[]) const {
    fbCresc->dumpObserved(observedOut);
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
