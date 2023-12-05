// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file interlevel.h

   @brief Manages the lazy repartitioning of the observation set.

   Splitting requires accessing the observations in sorted/grouped
   form.  Algorithms that do not attempt to split every node/predictor
   pair, such as Random Forest, can improve training speed by performing
   this updating (repartitioning) lazily.

   @author Mark Seligman
 */

#ifndef FRONTIER_INTERLEVEL_H
#define FRONTIER_INTERLEVEL_H

#include "algparam.h"
#include "path.h"
#include "partition.h"
#include "splitcoord.h"
#include "stagedcell.h"
#include "typeparam.h"

#include <deque>
#include <vector>
#include <map>

struct Ancestor {
  StagedCell& cell;
  unsigned int historyIdx;

  Ancestor(StagedCell& cell_,
	   unsigned int historyIdx_) :
    cell(cell_),
    historyIdx(historyIdx_) {
  }
};


/**
   @brief Manages definitions reaching the frontier.
 */
class InterLevel {
  const class PredictorFrame* frame;
  const PredictorT nPred; // Number of predictors.
  const PredictorT positionMask;
  const unsigned int levelShift;
  const IndexT bagCount;
  
  static constexpr double stageEfficiency = 0.15; // Work efficiency threshold.

  const class PredictorFrame* layout;
  const IndexT noRank; ///< inachievable rank value:  (re)staging.
  const class SampledObs* sampledObs;
  unique_ptr<class IdxPath> rootPath; // Root-relative IdxPath.
  vector<PathT> pathIdx;
  unsigned int level; // Zero-based tree depth.
  IndexT splitCount; // # nodes in the layer about to split.
  vector<Ancestor> ancestor; // Collection of ancestors to restage.
  unique_ptr<class ObsPart> obsPart;

  vector<vector<PredictorT>> stageMap; // Packed level, position.
  deque<unique_ptr<class ObsFrontier>> history; // Caches previous frontier layers.

  unique_ptr<class ObsFrontier> ofFront; ///< Current frontier, not in deque.

  
  /**
     @brief Rebuilds stage map for new frontier.

  */
  void reviseStageMap(const vector<class IndexSet>& frontierNodes);


  void pathHistory(const class IndexSet& iSet);


  /**
     @brief Derives mask sufficient to represent all offsets.

     @param nPred is the maximum predictor offset value.

     @return mask capable of identifying all values in [0, nPred].
   */
  static PredictorT getPositionMask(PredictorT nPred) {
    PredictorT bits = 2;
    while (bits <= nPred) // 'nPred' must be storable.
      bits <<= 1;
    return bits - 1;
  }
  

  /**
     @brief Derives shift value sufficient to accommodate offsets.
   */
  static unsigned int getLevelShift(PredictorT nPred) {
    unsigned int shiftVal = 1;
    while ((1ul << shiftVal) <= nPred) // 'nPred' must be storable.
      shiftVal++;

    return shiftVal;
  }

  bool isStaged(const SplitCoord& coord) const {
    return (stageMap[coord.nodeIdx][coord.predIdx] & positionMask) != nPred;
  }


public:

  /**
     @brief Class constructor.

     @param frame_ is the training frame.

     @param frontier_ tracks the frontier nodes.
  */
  InterLevel(const class PredictorFrame* frame,
	     const class SampledObs* sampledObs,
	     const class Frontier* frontier);

  
  /**
     @brief Class finalizer.
  */
  ~InterLevel() = default;


  /**
     @brief Passes through to frame method.
   */
  bool isFactor(PredictorT predIdx) const;


  /**
     @brief Prestages moribund rear history layers.

     @return count of rear layers suitable for popping.
   */
  unsigned int prestageRear();


  /**
     @brief Pushes first layer's path maps back to all back layers.
  */
  void backdate() const;

  
  /**
     @brief Sets root path successor and, if transitional, live path.
   */
  void rootSuccessor(IndexT sIdx,
		     PathT path,
		     IndexT destIdx);


  /**
     @brief Sets live path when layers remain.
   */
  void rootUpdate(IndexT sIdx,
		  PathT path,
		  IndexT destIdx);

  
  /**
     @brief Sets the root-relative live path.

     @param smIdx is SampleMap index for upcoming nonterminal.
   */
  void rootLive(IndexT sIdx,
		PathT path,
		IndexT smIdx);


  /**
     @brief Marks root-relative path as extinct.
   */
  void rootExtinct(IndexT rootIdx);


  /**
     @brief Marks root path as extinct.
   */
  void updateExtinct(IndexT rootIdx);


  /**
    @brief Interpolates splitting rank using observation bounds.

    @return fractional splitting "rank".
   */
  double interpolateRank(const class SplitNux& cand,
			 IndexT obsLeft,
			 IndexT obsRight) const;

  
  /**
     @brief Interpolates splitting rank involving residual.

     @return fractional splitting "rank".
   */  
  double interpolateRank(const class SplitNux& cand,
			 IndexT obsIdx,
			 bool residualLeft) const;


  /**
     @brief isImplicit is true iff this is a residual.

     @return code associated with a given observation index.
   */
  IndexT getCode(const class SplitNux& cand,
		 IndexT obsIdx,
		 bool isImplicit) const;

  
  
  ObsPart* getObsPart() const;


  bool preschedule(const SplitCoord& splitCoord);

  
  class ObsFrontier* getFront();


  /**
     @brief Appends a source cell to the restaging ancestor set.
   */
  void appendAncestor(StagedCell& scAnc,
		      unsigned int historyIdx);


  IndexT getNoRank() const {
    return noRank;
  }


  /**
     @brief Does not screen out singletons.
   */
  bool isStaged(const SplitCoord& coord,
		       unsigned int& stageLevel,
		       PredictorT& predPos) const {
    PredictorT packedIndex = stageMap[coord.nodeIdx][coord.predIdx];
    stageLevel = (packedIndex >> levelShift);
    predPos = (packedIndex & positionMask);
    return predPos != nPred;
  }


  /**
     @return position of staged coordinate.
   */
  PredictorT getStagedPosition(const SplitCoord& coord) const {
    return stageMap[coord.nodeIdx][coord.predIdx] & positionMask;
  }

  
  void setStaged(IndexT nodeIdx, 
		 PredictorT predIdx,
		 PredictorT offset) {
    stageMap[nodeIdx][predIdx] = (level << levelShift) | offset;
  }

  
  /**
     @brief Marks the specified cell as unsplitable.

     The unstaged placeholder value is sticky and persists through all
     successor nodes.
   */
  void delist(const SplitCoord& coord) {
    stageMap[coord.nodeIdx][coord.predIdx] = nPred;
  }


  PredictorT getNPred() const {
    return nPred;
  }


  IndexT getNSplit() const {
    return splitCount;
  }


  unsigned int getLevel() const {
    return level;
  }
  

  class ObsFrontier* getHistory(unsigned int del) const {
    return history[del].get();
  }


  /**
     @return base of indexed paths for a given predictor.
   */
  PathT* getPathBlock(PredictorT predIdx);
  

  IndexT* getIdxBuffer(const class SplitNux& nux) const;


  class Obs* getPredBase(const SplitNux& nux) const;


  /**
     @brief Intializes observations cells.
   */
  vector<unsigned int> stage();


  /**
     @brief Updates the data (observation) partition.
   */
  vector<unsigned int> restage();


  /**
     @brief Repartitions observations at a specified cell.

     @param mrra contains the coordinates of the originating cell.
   */
  unsigned int restage(Ancestor& ancestor);

  
  /**
     @brief Partitions or repartitions observations.

     @param frontier is the invoking Frontier.
   */
  CandType repartition(const class Frontier* frontier);


  /**
     @brief Updates subtree and pretree mappings from temporaries constructed
     during the overlap.  Initializes data structures for restaging and
     splitting the current layer of the subtree.
   */
  void overlap(const vector<class IndexSet>& frontierNodes,
	       const vector<class IndexSet>& frontierNext,
	       IndexT endIdx);


  /**
     @brief Terminates node-relative path an extinct index.  Also
     terminates subtree-relative path if currently live.

     @param nodeIdx is a node-relative index.

     @param stIdx is the subtree-relative index.
  */
  void relExtinct(IndexT nodeIdx,
		  IndexT stIdx);

  
  /**
     @brief Accessor for 'rootPath' field.
   */
  IdxPath* getRootPath() const {
    return rootPath.get();
  }


  /**
     @brief Destages all singletons among the newly-staged cells.

     @param splitCoord is the coordinate of a potential candidate.

     @param sc contains the rank and implicit count.
   */
  void destageSingleton(const SplitCoord& splitCoord,
			const StagedCell* sc);


  /**
     @brief Accessof for splitable node count in front layer.

     @return split count.
   */
  IndexT getSplitCount() const {
    return splitCount;
  }


  StagedCell*  getFrontCellAddr(const SplitCoord& coord);


  /**
     @param[out] cell is the staged cell, if any.

     @return true iff coordinate is staged.
   */
  bool isStaged(const SplitCoord& coord,
		StagedCell*& cell) const;
};


#endif

