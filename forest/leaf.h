// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file leaf.h

   @brief Records sample contents of leaf nodes.

   @author Mark Seligman
 */

#ifndef FOREST_LEAF_H
#define FOREST_LEAF_H

#include "typeparam.h"
#include "util.h"

#include <vector>

using namespace std;

class PreTree;
class Sampler;
class ResponseCtg;

/**
   @brief Rank and sample-counts associated with sampled rows.

   Client:  quantile inference.
 */
class RankCount {
  // When sampling is not weighted, the sample-count value typically
  // requires four bits or fewer.  Packing therefore accomodates rank
  // values well over 32 bits.
  PackedT packed; ///< Packed representation of rank and sample count.

  static unsigned int rightBits; ///< # bits occupied by rank value.
  static PackedT rankMask; ///< Mask unpacking the rank value.

public:

  /**
     @brief Invoked at Leaf construction, as needed.
   */
  static void setMasks(IndexT nObs) {
    rightBits = Util::packedWidth(nObs);
    rankMask = (1 << rightBits) - 1;
  }


  /**
     @brief Invoked at Sampler destruction.
   */
  static void unsetMasks() {
    rightBits = 0;
    rankMask = 0;
  }
  

  /**
     @brief Packs statistics associated with a response.

     @param rank is the rank of the response value.

     @param sCount is the number of times the observation was sampled.
   */
  void init(IndexT rank,
            IndexT sCount) {
    packed = rank | (sCount << rightBits);
  }

  
  IndexT getRank() const {
    return packed & rankMask;
  }


  IndexT getSCount() const {
    return packed >> rightBits;
  }
};


/**
   @brief Leaves are indexed by their numbering within the tree.
 */
struct Leaf {
  // Training only:
  vector<IndexT> indexCresc; ///< Sample indices within leaves.
  vector<IndexT> extentCresc; ///< Index extent, per leaf.
  
  // Post-training only:  extent, index maps fixed.
  const vector<vector<size_t>> extent; ///< # sample index entries per leaf, per tree.
  const vector<vector<vector<size_t>>> index; ///< sample indices per leaf, per tree.

  /**
     @brief Training factory.

     @param Sampler conveys observation count, to set static packing parameters.
   */
  static unique_ptr<Leaf> train(IndexT nObs);

  
  /**
     @brief Prediction factory.

     @param Sampler guides reading of leaf contents.

     @param extent gives the number of distinct samples, forest-wide.

     @param index gives sample positions.
  */
  static unique_ptr<Leaf> predict(const Sampler* sampler,
				  vector<vector<size_t>> extent,
				  vector<vector<vector<size_t>>> index);


  /**
     @brief Training constructor:  crescent structures only.
   */
  Leaf();

  
  /**
     @brief Post-training constructor:  fixed maps passed in.
   */
  Leaf(const Sampler* sampler,
       vector<vector<size_t>> extent_,
       vector<vector<vector<size_t>>> index_);

  
  /**
     @brief Resets static packing parameters.
   */
  ~Leaf();


  static Leaf unpack(const Sampler* sampler,
		     const double extent_[],
		     const double index_[]);


  static vector<vector<size_t>> unpackExtent(const Sampler* sampler,
					     const double extentNum[]);


  static vector<vector<vector<size_t>>> unpackIndex(const Sampler* sampler,
					     const vector<vector<size_t>>& extent,
					     const double numVal[]);


  /**
     @brief Indicates whether post-training leaf is empty.
     
     @return true iff the index vectors are unpopulated.
   */
  bool empty() const {
    return extent.empty();
  }

  
  /**
     @brief Copies terminal contents, if 'noLeaf' not specified.

     Training caches leaves in order of production.  Depth-first
     leaf numbering requires that the sample maps be reordered.
   */
  void consumeTerminals(const PreTree* pretree);


  /**
     @brief Enumerates the number of samples at each leaf's category.

     'probSample' is the only client.

     @return 3-d vector category counts, indexed by tree/leaf/ctg.
   */
  vector<vector<vector<size_t>>> countLeafCtg(const Sampler* sampler,
					      const ResponseCtg* response) const;


  /**
     @brief Count samples at each rank, per leaf, per tree:  regression.

     @param obs2Rank is the ranked training outcome.

     @return 3-d mapping as described.
   */
  vector<vector<vector<RankCount>>> alignRanks(const Sampler* sampler,
					       const vector<IndexT>& obs2Rank) const;


  /**
     @return # leaves at a given tree index.
   */
  size_t getLeafCount(unsigned int tIdx) const {
    return extent[tIdx].size();
  }


  const vector<IndexT>& getExtentCresc() const {
    return extentCresc;
  }


  const vector<IndexT>& getIndexCresc() const {
    return indexCresc;
  }
  
  /**
     @return vector of leaf extents for given tree.
   */
  const vector<size_t>& getExtents(unsigned int tIdx) const {
    return extent[tIdx];
  }


  /**
     @return vector of per-leaf index vectors for a given tree.
   */
  const vector<vector<size_t>>& getIndices(unsigned int tIdx) const {
    return index[tIdx];
  }
};

#endif
