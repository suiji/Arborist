// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file grove.h

   @brief Trains a block of trees.

   @author Mark Seligman
 */

#ifndef FOREST_GROVE_H
#define FOREST_GROVE_H

#include <string>
#include <vector>
#include <complex>

#include "decnode.h"
#include "typeparam.h"


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
   @brief Interface class for front end.  Holds simulation-specific parameters
   of the data and constructs forest, leaf and diagnostic structures.
*/
class Grove {
  static bool thinLeaves; ///< True iff leaves not cached.
  static unsigned int trainBlock; ///< Front-end defined buffer size. Unused.
  const IndexRange forestRange; ///< Coordinates within forest.
  const unique_ptr<struct NodeScorer> nodeScorer; ///< Belongs elsewhere.
  vector<double> predInfo; ///< E.g., Gini gain:  nPred.
  
  unique_ptr<NodeCresc> nodeCresc; ///< Crescent node block.
  unique_ptr<FBCresc> fbCresc; ///< Crescent factor-summary block.
  vector<double> scoresCresc; ///< Crescent score block.

public:

  /**
     @brief General constructor.
  */
  Grove(const class PredictorFrame* frame,
	const IndexRange& range);


  void train(const class PredictorFrame* frame,
	     const class Sampler* sampler,
	     struct Leaf* leaf);


  const vector<size_t>& getNodeExtents() const {
    return nodeCresc->getExtents();
  }


  const vector<double>& getScores() const {
    return scoresCresc;
  }
  
  void consumeTree(const vector<DecNode>& nodes,
		   const vector<double>& scores) {
    IndexT height = nodes.size();
    nodeCresc->consumeNodes(nodes, height);
    copy(scores.begin(), scores.begin() + height, back_inserter(scoresCresc));
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

  
  /**
     @brief Getter for splitting information values.

     @return reference to per-preditor information vector.
   */
  const vector<double> &getPredInfo() const {
    return predInfo;
  }


  static void init(bool thinLeaves_,
		   unsigned int trainBlock_);

 
  /**
     @brief Static de-initializer.
   */
  static void deInit();


  /**
     @brief Builds segment of decision forest for a block of trees.

     @param treeBlock is a vector of Sample, PreTree pairs.
  */
  void blockConsume(const vector<unique_ptr<class PreTree>> &treeBlock,
		    struct Leaf* leaf);


  /**
     @brief  Creates a block of root samples and trains each one.

     @return Wrapped collection of Sample, PreTree pairs.
  */
  vector<unique_ptr<class PreTree>> blockProduce(const class PredictorFrame* frame,
					   const class Sampler* sampler,
					   unsigned int treeStart,
					   unsigned int treeEnd);

  /**
     @brief Accumulates per-predictor information values from trained tree.
   */
  void consumeInfo(const vector<double>& info);


  struct NodeScorer* getNodeScorer() const {
    return nodeScorer.get();
  }
  
  /**
     @brief Getter for raw forest pointer.

  const Forest* getForest() const {
    return forest.get();//;//.get();
  }
  */

  size_t getNodeCount() const;
  void cacheNode(complex<double> complexOut[]) const;
  void cacheScore(double scoreOut[]) const;

  const vector<size_t>& getFacExtents() const;
  size_t getFactorBytes() const;

  /**
     @brief Dumps raw splitting values for factors.

     @param[out] rawOut outputs the raw bytes of factor split values.
  */
  void cacheFacRaw(unsigned char rawOut[]) const;

  
  void cacheObservedRaw(unsigned char observedOut[]) const;
};

#endif
