// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file obsfrontier.h

   @brief Tracks repartition definitions associated with a single frontier instance.

   Definitions cache the repartition state of a given splitting cell.
   Some algorithms, such as Random Forests, employ variable selection
   and do not require repartitioning of all cells at each frontier
   instance.  This allows repartitioning to be performed lazily and
   sparingly.

   @author Mark Seligman

 */

#ifndef OBS_OBSFRONTIER_H
#define OBS_OBSFRONTIER_H

#include "stagedcell.h"
#include "splitcoord.h"
#include "typeparam.h"

#include <vector>
#include <numeric>

/**
   @brief Caches previous frontier definitiions by layer.
 */
class ObsFrontier {
  const class Frontier* frontier;
  class InterLevel* interLevel;
  const PredictorT nPred; ///< Predictor count.
  const IndexT nSplit; ///< # splitable nodes at level.

  vector<IndexRange> node2Front;
  vector<IndexT> front2Node;

  vector<vector<StagedCell>> stagedCell; ///< Cell, node x predictor.
  IndexT stageCount; ///< # staged items.
  IndexT stageMax; ///< High watermark of stage count.
  IndexT runCount; ///< Total runs tracked.
  vector<IndexT> runValue; ///< Tracked run values.
  
  // layerIdx value is one less than distance to front.
  unsigned char layerIdx; ///< Zero-based deque offset, increments.

  // Recomputed:
  vector<class NodePath> nodePath; ///< Indexed by <node, predictor> pair.

  /**
     @brief Initializes a path from ancestor to front.
   */
  void pathInit(NodePath* pathBase,
		const class IndexSet& iSet);
  

  void updateLive(const class BranchSense& branchSense,
		  const class IndexSet& iSet,
		  const struct SampleMap& smNonterm,
		  struct SampleMap& smNext);


  /**
     @brief Updates terminals from extinct index sets.
   */
  void updateExtinct(const class IndexSet& iSet,
		     const struct SampleMap& smNonterm,
		     struct SampleMap& smTerminal);


  /**
     @brief Prestages all node indices referenced in range.
   */
  void prestageRange(const StagedCell& cell,
		     const IndexRange& range);


  /**
     @brief Enumerates the live cells.  Diagnostic.
   */
  IndexT countLive() const;

  
public:

  ObsFrontier(const class Frontier* frontier_,
	      class InterLevel* interLevel_);


  /**
     @brief Prestages an entire layer of eligible cells.

     @param ofCurrent is the current layer.
   */
  void prestageLayer(class ObsFrontier* ofCurrent);

  
  /**
     @brief Allocates all 'nPred' StagedCells for staging.
   */
  void prestageRoot(const class PredictorFrame* layout,
		    const class SampledObs* sampledObs);


  /**
     @brief Delists all live cells with an extinct node.     
   */
  void delistNode(IndexT nodeIdx);


  /**
     @brief Looks up the specified ancestor, prestages and appends to
     interLevel.
   */
  void prestageAncestor(ObsFrontier* ofFront,
			IndexT nodeIdx,
			PredictorT stagePosition);


  IndexT getStageCount() const {
    return stageCount;
  }


  /**
     @brief Computes percentage of full occupancy.

     'stageMax' should never be zero.
   */
  double stageOccupancy() const {
    return stageMax == 0 ? 0.0 : double(stageCount) / stageMax;
  }
  

  StagedCell getCell(IndexT nodeIdx,
		      PredictorT predPos) {
    return stagedCell[nodeIdx][predPos];
  }


  StagedCell* getCellAddr(IndexT nodeIdx,
			  PredictorT predPos) {
    return &stagedCell[nodeIdx][predPos];
  }


  void setFrontRange(const vector<IndexSet>& frontierNodes,
		     const vector<IndexSet>& frontierNext);

  
  /**
     @brief Builds to/from maps for a given node in the current level.

     Must be called in consecutive parIdx order.
   */
  void setFrontRange(const vector<class IndexSet>& frontierNext,
		     IndexT parIdx,
		     const IndexRange& range);


  /**
     @brief Getter for front range at a given split index.
   */
  inline IndexRange getFrontRange(IndexT splitIdx) const {
    return node2Front[splitIdx];
  }
  

  /**
     @brief Revises front ranges using current frontier.

     @param ofCurrent is the current frontier, about to enter deque.

     @param frontCount is the number of nodes in the new front.
  */
  void applyFront(const ObsFrontier* ofCurrent,
		  const vector<class IndexSet>& frontierNext);

  /**
     @brief Allocates the run values vector.
   */
  void runValues();

  
  /**
     @return number of staged cells (0 or 1).
   */
  unsigned int stage(PredictorT predIdx,
		     class ObsPart* obsPart,
		     const class PredictorFrame* layout,
		     const class SampledObs* sampledObs);


  /**
     @brief Repartitions previous ObsFrontier onto front.

     Precomputes path vector prior to restaging.  This is necessary in
     the case of dense ranks, as cell sizes are not derivable directly
     from index nodes.

     Decomposition into two paths adds ~5% performance penalty, but
     appears necessary for dense packing or for coprocessor loading.

     @param mrra is the ancestor residing in a history layer.

     @param ofFront is front ObsFrontier.

     @return count of delisted items:  <= # target items.
   */
  unsigned int restage(class ObsPart* obsPart,
		       const StagedCell& mrra,
		       ObsFrontier* ofFront) const;


  /**
     @brief Localizes copies of the paths to each index position.
   */
  vector<IndexT> pathRestage(class ObsPart* obsPart,
			     vector<IndexT>& preResidual,
			     vector<IndexT>& preNA,
			     const StagedCell& mrra) const;


  void restageRanks(const PathT* prePath,
		    ObsPart* obsPart,
		    vector<IndexT>& rankScatter,
		    const StagedCell& mrra,
		    vector<IndexT>& obsScatter,
		    vector<IndexT>& ranks) const;


  /**
     @brief Sets stage high watermark and adjust for extinction.

     Decrements stage count from vector computed in parallel.

     @param nExinct collects extinct counts.
   */
  void prune(const vector<unsigned int>& nExtinct) {
    stageMax = stageCount;
    stageCount -= accumulate(nExtinct.begin(), nExtinct.end(), 0);
  }


  /**
     @brief Delists cell and crements stage count.
   */
  inline void delist(StagedCell& cell) {
    cell.delist();
    stageCount--;
  }


  /**
   @brief Sets the packed offsets for each successor.  Relies on Swiss Cheese
   index numbering ut prevent cell boundaries from crossing.

   @param pathCount inputs the counts along each reaching path.

   @param dfCurrent is the current ObsFrontier.
 */
  vector<IndexT> packTargets(ObsPart* obsPart,
			     const StagedCell& mrra,
			     vector<StagedCell*>& tcp) const;


  /**
     @brief As above, but with additional value scatter vector.
   */
  vector<IndexT> packTargets(ObsPart* obsPart,
			     const StagedCell& mrra,
			     vector<StagedCell*>& tcp,
			     vector<IndexT>& valScatter) const;


  /**
     @brief Dispatches sample map update according to terminal/nonterminal.
   */
  void updateMap(const class IndexSet& iSet,
		 const class BranchSense& branchSense,
		 const struct SampleMap& smNonterm,
		 struct SampleMap& smTerminal,
		 struct SampleMap& smNext);


  /**
     @brief Shifts a value by the number of back-levels to compensate for
     effects of binary branching.

     @param val is the value to shift.

     @return shifted value.
   */  
  inline IndexT backScale(IndexT idx) const {
    return idx << (unsigned int) (layerIdx + 1);
  }


  /**
     @brief Produces mask approprate for level:  lowest 'del' bits high.

     @return bit mask value.
   */
  inline unsigned int pathMask() const {
    return backScale(1) - 1;
  }
  

  inline PredictorT getNPred() const {
    return nPred;
  }

  
  inline IndexT getNSplit() const {
    return nSplit;
  }
};

#endif
