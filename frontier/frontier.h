// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file frontier.h

   @brief Partitions tree frontier, typically breadth-first.

   @author Mark Seligman

 */

#ifndef PARTITION_FRONTIER_H
#define PARTITION_FRONTIER_H

#include "samplemap.h"
#include "pretree.h"
#include "indexset.h"
#include "typeparam.h"

#include <algorithm>
#include <vector>


/**
   @brief The index sets associated with nodes at a single subtree level.
 */
class Frontier {
  static unsigned int totLevels;
  const class TrainFrame* frame;
  const unique_ptr<class Sample> sample;
  const IndexT bagCount;
  const PredictorT nCtg;

  vector<IndexSet> frontierNodes;
  unique_ptr<class DefMap> defMap;

  unique_ptr<PreTree> pretree; // Augmented per frontier.
  
  SampleMap smTerminal; // Persistent terminal sample mapping:  crescent.
  SampleMap smNonterm; // Current nonterminal mapping.

  unique_ptr<class SplitFrontier> splitFrontier; // Per-level.

  /**
     @brief Determines splitability of frontier nodes just split.

     @param indexSet holds the index-set representation of the nodes.
   */
  SampleMap surveySplits(vector<IndexSet>& indexSet);

  void registerSplit(IndexSet& iSet,
		    SampleMap& smNext);
  
  void registerTerminal(IndexSet& iSet);
  void registerNonterminal(IndexSet& iSet,
			   SampleMap& smNext);


  /**
     @brief Applies splitting results to new level.

     @param level is the current zero-based level.

  */
  vector<IndexSet> splitDispatch(const class BranchSense* branchSense);


  /**
     @brief Produces frontier nodes for next level.
   */
  vector<IndexSet> produce() const;

  
  /**
     @brief Resets parameters for upcoming levl.

     @param splitNext is the number of splits in the new level.
  */
  void reset(IndexT splitNext);


 public:


  /**
     @brief Initializes static invariants.

     @param minNode_ is the minimum node size for splitting.

     @param totLevels_ is the maximum number of levels to evaluate.
  */
  static void immutables(unsigned int totLevels);


  /**
     @brief Resets statics to default values.
  */
  static void deImmutables();


  /**
     @brief Per-tree constructor.  Sets up root node for level zero.
  */
  Frontier(const class TrainFrame* frame_,
           unique_ptr<class Sample> sample_);

  
  /**
    @brief Trains one tree.

    @param summaryFrame contains the predictor type mappings.

    @param sample contains the bagging summary.

    @return trained pretree object.
  */
  static unique_ptr<class PreTree> oneTree(const class TrainFrame* frame,
					   class Sampler* sampler,
					   unsigned int tIdx);


  /**
     @brief Drives breadth-first splitting.

     Assumes root node and attendant per-tree data structures have been initialized.
     Parameters as described above.
  */
  unique_ptr<class PreTree> levels();


  /**
     @brief Marks frontier nodes as unsplitable for graceful termination.

     @param level is the zero-based tree depth.
   */
  void earlyExit(unsigned int level);


  /**
     @brief Updates both index set and pretree states for a set of simple splits.
   */
  void updateSimple(const vector<class SplitNux>& nuxMax,
		    class BranchSense* branchSense);


  /**
     @brief Updates only pretree states for a set of compound splits.
   */
  void updateCompound(const vector<vector<class SplitNux>>& nuxMax);


  void setScore(IndexT splitIdx) const;


  const IndexSet& getNode(IndexT splitIdx) const {
    return frontierNodes[splitIdx];
  }

  
  /**
     @return pre-tree index associated with node.
   */
  IndexT getPTId(const MRRA& mrra) const {
    return frontierNodes[mrra.splitCoord.nodeIdx].getPTId();
  }
  

  /**
     @brief PreTree pass-through to obtain successor index.

     @param ptId is the parent pretree index.

     @param senseTrue is true iff true branch sense requested.

     @return successor pre-tree index.
   */
  IndexT getPTIdSucc(IndexT ptId,
                     bool senseTrue) const;


  /**
     @brief DefMap pass-through to register reaching path.

     @param splitIdx is the level-relative node index.

     @param parIdx is the parent node's index.
   */
  void reachingPath(const IndexSet& iSet,
                    IndexT parIdx) const;


  /**
     @brief Updates the split/path/pretree state of an extant index based on
     its position in the next level (i.e., left/right/extinct).

     @param stPath is a subtree-relative path.
  */
  void stReindex(const class BranchSense* branchSense,
                 IndexT splitNext,
                 IndexT chunkStart,
                 IndexT chunkNext);


  /**
     @brief Visits all live indices, so likely worth parallelizing.

     @param[out] ctgSum outputs the per-category sum of responses, per node.

     @return category-summed squares of responses, per node.
  */
  vector<double> sumsAndSquares(vector<vector<double> >& ctgSum);

  
  /**
     @brief Passes through to IndexSet method.

     @param splitIdx could also be looked up from candV, if nonempty.

     @param[out] argMax most informative split associated with node, if any.

     @param candV contains splitting candidates associated with split index.
   */
  void candMax(IndexT splitIdx,
	       class SplitNux& argMax,
	       const vector<class SplitNux>& candV) const {
    frontierNodes[splitIdx].candMax(candV, argMax);
  }


  /**
     @brief Obtains the IndexRange for a splitting candidate's location.

     @param mrra contains candidate's coordinate.

     @return index range of referenced split coordinate.
   */
  IndexRange getBufRange(const MRRA& mrra) const {
    return frontierNodes[mrra.splitCoord.nodeIdx].getBufRange();
  }


  auto getDefMap() const {
    return defMap.get();
  }


  auto getFrame() const {
    return frame;
  }
  

  /**
     @brief Getter for # of distinct in-bag samples.

     @return bagCount value.
   */
  inline auto getBagCount() const {
    return bagCount;
  }


  /**
     @brief Getter for # categories in response.
   */
  inline auto getNCtg() const {
    return nCtg;
  }

  
  /**
     @brief Accessor for count of splitable sets.
   */
  inline IndexT getNSplit() const {
    return frontierNodes.size();
  }


  /**
     @brief Accessor for sum of sampled responses over set.

     @param splitIdx is the level-relative index of a set.

     @return index set's sum value.
   */
  inline auto getSum(IndexT splitIdx) const {
    return frontierNodes[splitIdx].getSum();
  }


  /**
     @brief As above, but parametrized by candidate location.
   */
  inline auto getSum(const MRRA& mrra) const {
    return getSum(mrra.splitCoord.nodeIdx);
  }


  /**
     @brief Accessor for count of sampled responses over set.
   */
  inline auto getSCount(IndexT splitIdx) const {
    return frontierNodes[splitIdx].getSCount();
  }


  inline auto getSCount(const MRRA& mrra) const {
    return getSCount(mrra.splitCoord.nodeIdx);
  }


  /**
     @brief Accessor for count of sampled responses over set.
   */
  inline auto getSCountSucc(IndexT splitIdx,
			    bool sense) const {
    return frontierNodes[splitIdx].getSCountSucc(sense);
  }


  inline auto getSCountSucc(const MRRA& mrra,
			    bool sense) const {
    return getSCountSucc(mrra.splitCoord.nodeIdx, sense);
  }


  /**
     @brief Accessor for count of sampled responses over set.
   */
  inline auto getSumSucc(IndexT splitIdx,
			    bool sense) const {
    return frontierNodes[splitIdx].getSumSucc(sense);
  }


  inline auto getSumSucc(const MRRA& mrra,
			    bool sense) const {
    return getSumSucc(mrra.splitCoord.nodeIdx, sense);
  }


  /**
     @brief Accessor for count of disinct indices over set.
   */
  inline auto getExtent(IndexT splitIdx) const {
    return frontierNodes[splitIdx].getExtent();
  }
  

  IndexRange getNontermRange(const IndexSet& iSet) const {
    return smNonterm.range[iSet.getSplitIdx()];
  }

  
  /**
     @brief Indicates whether index set is inherently unsplitable.
   */
  inline bool isUnsplitable(IndexT splitIdx) const {
    return frontierNodes[splitIdx].isUnsplitable();
  }

  
  /**
     @brief Dispatches consecutive node-relative indices to frontier map for
     final pre-tree node assignment.
  */
  void relExtinct(const IndexSet& iSet);
};

#endif
