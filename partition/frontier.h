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

#include <vector>


/**
   @brief Enumerates split characteristics over a trained frontier.
 */
struct SplitSurvey {
  IndexT leafCount; // Number of terminals in this layer.
  IndexT idxLive; // Extent of live buffer indices.
  IndexT splitNext; // Number of splitable nodes in next layer.
  IndexT idxMax; // Maximum index.

  SplitSurvey() :
    leafCount(0),
    idxLive(0),
    splitNext(0),
    idxMax(0){
  }


  /**
     brief Imputes the number of successor nodes, including pseudosplits.
   */
  IndexT succCount(IndexT splitCount) const {
    IndexT leafNext = 2 * (splitCount - leafCount) - splitNext;
    return splitNext + leafNext + leafCount;
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
  unique_ptr<class DefMap> defMap;
  bool nodeRel; // Whether level uses node-relative indexing:  sticky.
  IndexT idxLive; // Total live indices.
  IndexT liveBase; // Accumulates live index offset.
  IndexT extinctBase; // Accumulates extinct index offset.
  IndexT succLive; // Accumulates live indices for upcoming level.
  IndexT succExtinct; // " " extinct "
  vector<IndexT> relBase; // Node-to-relative index.
  vector<IndexT> succBase; // Overlaps, then moves to relBase.
  vector<IndexT> rel2ST; // Node-relative mapping to subtree index.
  vector<IndexT> rel2PT; // Node-relative mapping to pretree index.
  vector<IndexT> st2Split; // Subtree-relative mapping to split index.
  vector<IndexT> st2PT; // Subtree-relative mapping to pretree index.
  unique_ptr<PreTree> pretree; // Augmented per frontier.
  
  
  SplitSurvey surveySet(vector<IndexSet>& indexSet);

  

  /**
     @brief Applies splitting results to new level.

     @param level is the current zero-based level.

  */
  vector<IndexSet> splitDispatch(const class BranchSense* branchSense,
				 unsigned int level);

  /**
     @brief Establishes splitting parameters for next frontier level.
   */
  SplitSurvey nextLevel(unsigned int level);
  

  /**
     @brief Resets parameters for upcoming levl.

     @param splitNext is the number of splits in the new level.
  */
  void reset(IndexT splitNext);


  /**
     @brief Reindexes by level modes: node-relative, subtree-relative, mixed.

     Parameters as above.
   */
  void reindex(const class BranchSense* branchSense,
	       const SplitSurvey& survey);

  
  /**
     @brief Produces new level's index sets and dispatches extinct nodes to pretree frontier.

     Parameters as described above.

     @return next level's splitable index set.
  */
  vector<IndexSet> produce(IndexT splitNext);


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
  Frontier(const class TrainFrame* frame,
           const class Sample* sample);

  ~Frontier();

  
  /**
    @brief Trains one tree.

    @param summaryFrame contains the predictor type mappings.

    @param sample contains the bagging summary.

    @return trained pretree object.
  */
  static unique_ptr<class PreTree> oneTree(const class TrainFrame* frame,
					   const class Sample* sample);


  /**
     @brief Drives breadth-first splitting.

     Assumes root node and attendant per-tree data structures have been initialized.
     Parameters as described above.
  */
  unique_ptr<class PreTree> levels(const class Sample* sample);
  

  /**
     @brief Builds index base offsets to mirror crescent pretree level.

     @param extent is the count of the index range.

     @param offOut outputs the node-relative starting index.  Should not
     exceed 'idxExtent', the live high watermark of the previous level.

     @param terminal is true iff successor is known a priori to be terminal.

     @return successor index count.
  */
  IndexT idxSucc(IndexT extent,
                 IndexT &outOff,
                 bool terminal = false);


  IndexT getPTId(const PreCand& preCand) const {
    return indexSet[preCand.splitCoord.nodeIdx].getPTId();
  }
  

  /**
     @brief PreTree pass-through to obtain successor index.

     @param ptId is the parent pretree index.

     @param senseTrue is true iff true branch sense requested.

     @return successor index.
   */
  IndexT getPTIdSucc(IndexT ptId,
                     bool senseTrue) const;


  /**
     @brief Obtains pretree indices for true and false branch targets.
   */
  void getPTIdTF(IndexT ptId,
                 IndexT& ptTrue,
                 IndexT& ptFalse) const;

  /**
     @brief DefMap pass-through to register reaching path.

     @param splitIdx is the level-relative node index.

     @param parIdx is the parent node's index.

     @param bufRange is the subsumed buffer range.

     @param relBase is the index base.

     @param path is the inherited path.
   */
  void reachingPath(IndexT splitIdx,
                    IndexT parIdx,
                    const IndexRange& bufRange,
                    IndexT relBase,
                    unsigned int path) const;
  
  /**
     @brief Drives node-relative re-indexing.
   */
  void nodeReindex(const class BranchSense* branchSense);

  /**
     @brief Subtree-relative reindexing:  indices randomly distributed
     among nodes (i.e., index sets).
  */
  void stReindex(const class BranchSense* branchSense,
		 IndexT splitNext);

  /**
     @brief Updates the split/path/pretree state of an extant index based on
     its position in the next level (i.e., left/right/extinct).

     @param stPath is a subtree-relative path.
  */
  void stReindex(const class BranchSense* branchSense,
		 class IdxPath *stPath,
                 IndexT splitNext,
                 IndexT chunkStart,
                 IndexT chunkNext);

  /**
     @brief As above, but initializes node-relative mappings for subsequent
     levels.  Employs accumulated state and cannot be parallelized.
  */
  void transitionReindex(const class BranchSense* branchSense,
			 IndexT splitNext);

  /**
     @brief Updates the mapping from live relative indices to associated
     PreTree indices.

     @return corresponding subtree-relative index.
  */
  IndexT relLive(IndexT relIdx,
                       IndexT targIdx,
                       IndexT path,
                       IndexT base,
                       IndexT ptIdx);
  /**
     @brief Translates node-relative back to subtree-relative indices on 
     terminatinal node.

     @param relIdx is the node-relative index.

     @param ptId is the pre-tree index of the associated node.
  */
  void relExtinct(IndexT relIdx,
                  IndexT ptId);

  
  /**
     @brief Visits all live indices, so likely worth parallelizing.

     @param[out] ctgSum outputs the per-category sum of responses, per node.

     @return category-summed squares of responses, per node.
  */
  vector<double> sumsAndSquares(vector<vector<double> >& ctgSum);

  /**
     @brief Getter for IndexRange at a given coordinate.

     @param preCand contains the splitting candidate's coordinate.

     @return index range of referenced split coordinate.
   */
  IndexRange getBufRange(const PreCand& preCand) const;


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


  inline auto getNCtg() const {
    return nCtg;
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


  inline auto getSum(const PreCand& preCand) const {
    return getSum(preCand.splitCoord.nodeIdx);
  }


  /**
     @brief Accessor for count of sampled responses over set.
   */
  inline auto getSCount(IndexT splitIdx) const {
    return indexSet[splitIdx].getSCount();
  }


  inline auto getSCount(const PreCand& preCand) const {
    return getSCount(preCand.splitCoord.nodeIdx);
  }


  /**
     @brief Accessor for count of disinct indices over set.
   */
  inline auto getExtent(IndexT splitIdx) const {
    return indexSet[splitIdx].getExtent();
  }
  

  /**
     @brief Accessor for relative base of split.
   */
  inline auto getRelBase(IndexT splitIdx) const {
    return relBase[splitIdx];
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
  void relExtinct(IndexT relBase,
                  IndexT extent,
                  IndexT ptId) {
    for (IndexT relIdx = relBase; relIdx < relBase + extent; relIdx++) {
      relExtinct(relIdx, ptId);
    }
  }


  /**
     @brief Reconciles remaining live node-relative indices.
  */
  void relFlush() {
    if (nodeRel) {
      for (IndexT relIdx = 0; relIdx < idxLive; relIdx++) {
        relExtinct(relIdx, rel2PT[relIdx]);
      }
    }
  }
};

#endif
