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

#include "pretree.h"
#include "indexset.h"
#include "typeparam.h"

#include <algorithm>
#include <vector>
#include <numeric>


/**
   @brief Maps to and from sample indices and tree nodes.

   Easy access to node contents simplifies the task of scoring both terminals
   and nonterminals.  Nonterminal scores provide a prediction for premature
   termination, as in the case of missing observations.

   Nonterminal component is maintained via a double-buffer scheme, updated
   following splitting.  The update performs a stable partition to improve
   latency.  The buffer initially lists all sample indices, but continues to
   shrink as terminal nodes absorb the contents.

   Terminal component is initially empty, but continues to grow as nonterminal
   contents are absorbed.

   Extent vectors record the number of sample indices associated with each node.
 */
struct SampleMap {
  vector<IndexT> indices;
  vector<IndexRange> range;
  vector<IndexT> ptIdx;
  IndexT maxExtent; // Tracks width of node-relative indices.

  /**
     @brief Constructor with optional index count.
   */
  SampleMap(IndexT nIdx = 0) :
    indices(vector<IndexT>(nIdx)),
    range(vector<IndexRange>(0)),
    ptIdx(vector<IndexT>(0)),
    maxExtent(0) {
  }


  IndexT getEndIdx() const {
    return range.empty() ? 0 : range.back().getEnd();
  }


  void addNode(IndexT extent,
	       IndexT ptId) {
    maxExtent = max(maxExtent, extent);
    range.emplace_back(getEndIdx(), extent);
    ptIdx.push_back(ptId);
  }

  
  IndexT* getWriteStart(IndexT idx) {
    return &indices[range[idx].getStart()];
  }
  
  
  IndexT getNodeCount() const {
    return range.size();
  }
};


/**
   @brief The index sets associated with nodes at a single subtree level.
 */
class Frontier {
  static unsigned int totLevels;
  const class TrainFrame* frame;
  vector<IndexSet> indexSet;
  const IndexT bagCount;
  const PredictorT nCtg;
  unique_ptr<class DefFrontier> defMap;
  bool nodeRel; // Whether level uses node-relative indexing:  sticky.

  unique_ptr<PreTree> pretree; // Augmented per frontier.
  
  SampleMap smTerminal; // Persistent terminal sample mapping:  crescent.
  IndexT terminalNodes; // Beginning # terminal nodes at level.

  SampleMap smNonterm; // Current nonterminal mapping.

  const vector<IndexT> recoverSt2PT() const;

  /**
     @brief Determines splitability of frontier nodes just split.

     @param indexSet holds the index-set representation of the nodes.
   */
  SampleMap surveySet(vector<IndexSet>& indexSet);

  void surveySplit(IndexSet& iSet,
		   SampleMap& smNext);
  
  void registerTerminal(IndexSet& iSet);
  void registerNonterminal(IndexSet& iSet,
			   SampleMap& smNext);
  

  /**
     @brief Dispatches sample map update according to terminal/nonterminal.
   */
  void updateMap(IndexSet& iSet,
		 const class BranchSense* branchSense,
		 SampleMap& smNext,
		 bool transitional);


  /**
     @brief Applies splitting results to new level.

     @param level is the current zero-based level.

  */
  vector<IndexSet> splitDispatch(const class BranchSense* branchSense);


  /**
     @brief Establishes splitting parameters for next frontier level.
   */
  void nextLevel(const class BranchSense*,
		 SampleMap& smNext);
  

  /**
     @brief Resets parameters for upcoming levl.

     @param splitNext is the number of splits in the new level.
  */
  void reset(IndexT splitNext);


  /**
     @brief Produces new level's node and marks unsplitable nodes.
  */
  vector<IndexSet> produce();


 public:
  /**
     @brief Updates terminals from extinct index sets.
   */
  void updateExtinct(const IndexSet& iSet,
		     bool transitional);


  /**
     @brief Updates terminals and nonterminals from live index sets.
   */
  void updateLive(const class BranchSense* branchSense,
		  const IndexSet& iSet,
		  SampleMap& smNext,
		  bool transitional);


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
  Frontier(const class TrainFrame* frame,
           const class Sample* sample);

  
  /**
    @brief Trains one tree.

    @param summaryFrame contains the predictor type mappings.

    @param sample contains the bagging summary.

    @return trained pretree object.
  */
  static unique_ptr<class PreTree> oneTree(const class TrainFrame* frame,
					   class Sampler* sampler);


  /**
     @brief Drives breadth-first splitting.

     Assumes root node and attendant per-tree data structures have been initialized.
     Parameters as described above.
  */
  unique_ptr<class PreTree> levels(const class Sample* sample);


  /**
     @brief Marks frontier nodes as unsplitable for graceful termination.

     @param level is the zero-based tree depth.
   */
  void earlyExit(unsigned int level);


  /**
     @brief Updates both index set and pretree states for a set of simple splits.
   */
  void updateSimple(const class SplitFrontier* sf,
		    const vector<class SplitNux>& nuxMax,
		    class BranchSense* branchSense);


  /**
     @brief Updates only pretree states for a set of compound splits.
   */
  void updateCompound(const class SplitFrontier* sf,
		      const vector<vector<class SplitNux>>& nuxMax);

  /**
     @return pre-tree index associated with node.
   */
  IndexT getPTId(const MRRA& mrra) const {
    return indexSet[mrra.splitCoord.nodeIdx].getPTId();
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
     @brief DefFrontier pass-through to register reaching path.

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
    indexSet[splitIdx].candMax(candV, argMax);
  }


  /**
     @brief Obtains the IndexRange for a splitting candidate's location.

     @param mrra contains candidate's coordinate.

     @return index range of referenced split coordinate.
   */
  IndexRange getBufRange(const MRRA& mrra) const {
    return indexSet[mrra.splitCoord.nodeIdx].getBufRange();
  }


  auto getDefFrontier() const {
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


  bool nodeRelative() const {
    return nodeRel;
  }

  
  /**
     @brief Accessor for count of splitable sets.
   */
  inline IndexT getNSplit() const {
    return indexSet.size();
  }


  /**
     @brief Accessor for sum of sampled responses over set.

     @param splitIdx is the level-relative index of a set.

     @return index set's sum value.
   */
  inline auto getSum(IndexT splitIdx) const {
    return indexSet[splitIdx].getSum();
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
    return indexSet[splitIdx].getSCount();
  }


  inline auto getSCount(const MRRA& mrra) const {
    return getSCount(mrra.splitCoord.nodeIdx);
  }


  /**
     @brief Accessor for count of sampled responses over set.
   */
  inline auto getSCountSucc(IndexT splitIdx,
			    bool sense) const {
    return indexSet[splitIdx].getSCountSucc(sense);
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
    return indexSet[splitIdx].getSumSucc(sense);
  }


  inline auto getSumSucc(const MRRA& mrra,
			    bool sense) const {
    return getSumSucc(mrra.splitCoord.nodeIdx, sense);
  }


  /**
     @brief Accessor for count of disinct indices over set.
   */
  inline auto getExtent(IndexT splitIdx) const {
    return indexSet[splitIdx].getExtent();
  }
  

  IndexRange getNontermRange(const IndexSet& iSet) const {
    return smNonterm.range[iSet.getSplitIdx()];
  }

  
  /**
     @brief Indicates whether index set is inherently unsplitable.
   */
  inline bool isUnsplitable(IndexT splitIdx) const {
    return indexSet[splitIdx].isUnsplitable();
  }

  
  /**
     @brief Dispatches consecutive node-relative indices to frontier map for
     final pre-tree node assignment.
  */
  void relExtinct(const IndexSet& iSet);


  /**
     @brief Reconciles remaining live node-relative indices.
  */
  void relFlush();
};

#endif
