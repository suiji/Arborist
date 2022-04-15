// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file defmap.h

   @brief Manages the lazy repartitioning of the observation set.

   Splitting requires accessing the observations in sorted/grouped
   form.  Algorithms that do not attempt to split every node/predictor
   pair, such as Random Forest, can improve training speed by performing
   this updating (repartitioning) lazily.

   @author Mark Seligman
 */

#ifndef FRONTIER_DEFMAP_H
#define FRONTIER_DEFMAP_H


#include <deque>
#include <vector>
#include <map>

#include "mrra.h"
#include "splitcoord.h"
#include "stagecount.h"
#include "typeparam.h"


/**
   @brief Minimal information needed to define a splitting pre=candidate.
 */
struct PreCand {
  MRRA mrra; // delIdx implicitly zero, but buf-bit needed.
  StageCount stageCount; // Shared between candidate and accumulator, if cand.
  uint32_t randVal; // Arbiter for tie-breaking and the like.
  
  /**
     @brief MRRA component initialized at construction, StageCount at (re)staging.
   */
  PreCand(const SplitCoord& splitCoord,
	  unsigned int bufIdx,
	  uint32_t randVal_) :
    mrra(MRRA(splitCoord, bufIdx, 0)),
    randVal(randVal_) {
  }

  
  void setStageCount(const StageCount& sc) {
    stageCount = sc;
  }


  bool isSingleton() const {
    return stageCount.isSingleton();
  }

  
  /**
     @brief Checks whether StageCount member has been initialized.

     Testing only.
   */
  bool isInitialized() const {
    return stageCount.isInitialized();
  }
};


/**
   @brief Manages definitions reaching the frontier.
 */
class DefMap {
  const PredictorT nPred; // Number of predictors.
  class Frontier* frontier;
  const IndexT bagCount;
  
  static constexpr double efficiency = 0.15; // Work efficiency threshold.

  unique_ptr<class IdxPath> rootPath; // Root-relative IdxPath:  EXIT.
  IndexT splitPrev; // # nodes in previous layer.
  IndexT splitCount; // # nodes in the layer about to split.
  const class Layout* layout;
  const PredictorT nPredDense; // Number of predictors using dense indexing.
  const vector<IndexT> denseIdx; // # Compressed mapping to dense offsets.
  vector<MRRA> ancestor; // Collection of ancestors to restage.
  unique_ptr<class ObsPart> obsPart;

  vector<unsigned int> history; // Current layer's history.
  vector<unsigned int> historyPrev; // Previous layer's history:  accum.
  vector<unsigned char> layerDelta; // # layers back split was defined.
  vector<unsigned char> deltaPrev; // Previous layer's delta:  accum.
  deque<unique_ptr<class DefFrontier> > layer; // Caches layers tracked by history.
  vector<vector<PreCand>> preCand; // Restageable, possibly splitable, coordinates.


  /**
     @brief Increments reaching layers for all pairs involving node.

     @param splitIdx is the index of a splitting node w.r.t. current layer.

     @param parIdx is the index of the parent w.r.t. previous layer.
   */
  inline void inherit(IndexT splitIdx,
		      IndexT parIdx) {
    unsigned char *colCur = &layerDelta[splitIdx * nPred];
    unsigned char *colPrev = &deltaPrev[parIdx * nPred];
    for (PredictorT predIdx = 0; predIdx < nPred; predIdx++) {
      colCur[predIdx] = colPrev[predIdx] + 1;
    }
  }


  /**
     @brief Passes ObsPart through to Sample method.
   */
  void stage(const class SampleObs* sample);


 public:

  /**
     @brief Class constructor.

     @param frame_ is the training frame.

     @param frontier_ tracks the frontier nodes.
  */
  DefMap(const class TrainFrame* frame,
	      class Frontier* frontier);

  
  /**
     @brief Class finalizer.
  */
  ~DefMap();


  /**
     @brief Establishes splitting parameters for next frontier level.
   */
  void nextLevel(const class BranchSense*,
		 const class SampleMap& smNonterm,
		 class SampleMap& smTerminal,
		 class SampleMap& smNext);
  

  /**
     @brief Pushes first layer's path maps back to all back layers
     employing node-relative indexing.
  */
  void backdate() const;

  
  /**
     @brief Sets root path successor and, if transitional, live path.
   */
  void rootSuccessor(IndexT sIdx,
		     PathT path,
		     IndexT destIdx);


  /**
     @brief Sets live path when root-relative layers remain.
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
     @brief Marks root path as extinct iff back layer is root-relative.
   */
  void updateExtinct(IndexT rootIdx);

  
  const class ObsPart* getObsPart() const;

  
  /**
     @brief Dense offsets maintained separately, as a special case.

     @return offset strided by 'nPredDense'.
   */
  inline IndexT denseOffset(const SplitCoord& splitCoord) const {
    return splitCoord.nodeIdx * nPredDense + denseIdx[splitCoord.predIdx];
  }


  inline IndexT denseOffset(const MRRA& cand) const {
    return denseOffset(cand.splitCoord);
  }


  inline PredictorT getNPred() const {
    return nPred;
  }


  inline IndexT getNSplit() const {
    return splitCount;
  }


  PredictorT getNPredDense() const {
    return nPredDense;
  }


  class DefFrontier* getLayer(unsigned int del) const {
    return layer[del].get();
  }


  /**
     @brief Passes through to Frontier method.

     @return true iff indexed split is not splitable.
   */
  bool isUnsplitable(IndexT splitIdx) const;


  /**
     @brief Rebuilds the precandidate vector using CandT method.
   */
  void setPrecandidates(const class SampleObs* sample,
			unsigned int level);

  
  const vector<vector<PreCand>>& getPrecand() const {
    return preCand;
  }


  /**
     @brief Gleans singletons from precandidate set.

     @return vector of candidates.
   */
  vector<class SplitNux> getCandidates(const class SplitFrontier* sf) const;
  

  /**
     @brief Clears ancestor list and lazily erases rear layers.

     Reaching layers must persist through restaging ut allow path lookup.
     @param flushCount is the number of rear layers to erase.
  */
  void clearDefs(unsigned int flushCount);


  /**
     @brief Reports schedulability of preflushed coordinate.
     
     @param[out] outputs the cell's buffer index, if splitable.

     @return true iff cell at coordinate is splitable.
  */
  bool preschedulable(const SplitCoord& splitCoord,
		      unsigned int& bufIdx) const;


  /**
     @brief As above, but discovers buffer via lookup.
   */
  bool preschedule(const SplitCoord& splitCoord,
		   double dRand);


  /**
     @brief Extracts the 32 lowest-order mantissa bits of a double-valued
     random variate.

     The double-valued variants passed are used by the caller to arbitrate
     variable sampling and are unlikely to rely on more than the first
     few mantissa bits.  Hence using the low-order bits to arbitrate other
     choices is unlikely to introduce spurious correlations.
   */
  inline static unsigned int getRandLow(double rVal) {
    union { double d; uint32_t ui[2]; } u = {rVal};
    
    return u.ui[0];
  }



  
  /**
     @brief Passes through to front layer.
   */
  bool isSingleton(const MRRA& defCoord) const;


  /**
     @brief Flips source bit if a definition reaches to current layer.
  */
  void addDef(const MRRA& splitCoord,
              bool singleton);


  /**
     @brief Passes through to front layer.
   */
  void adjustRange(const MRRA& preCand,
		   IndexRange& idxRange) const;


  IndexT* getIdxBuffer(const class SplitNux* nux) const;

  
  class ObsCell* getPredBase(const SplitNux* nux) const;

  
  /**
     @brief Passes through to front layer.
   */
  IndexT getImplicitCount(const MRRA& preCand) const;


  /**
     @brief Appends restaged ancestor.
   */
  void appendAncestor(const MRRA& mrra);


  /**
     @brief Updates the data (observation) partition.
   */
  void restage();



  /**
     @brief Repartitions observations at a specified cell.

     @param mrra contains the coordinates of the originating cell.
   */
  void restage(const MRRA& mrra) const;
  
  /**
     @brief Updates subtree and pretree mappings from temporaries constructed
     during the overlap.  Initializes data structures for restaging and
     splitting the current layer of the subtree.
  */
  void overlap(const class SampleMap& smNonterm);



  /**
     @brief Consumes all fields from a node relevant to restaging.

     @param iSet is the node, in index set form.

     @param parent the node's parent.
  */
  void reachingPath(const class IndexSet& iSet,
		    const class IndexSet& parent);


  /**
     @brief Flushes non-reaching definitions as well as those about
     to fall off the layer deque.

     @return count of layers to flush.
  */
  unsigned int flushRear();


  /**
     @brief Terminates node-relative path an extinct index.  Also
     terminates subtree-relative path if currently live.

     @param nodeIdx is a node-relative index.

     @param stIdx is the subtree-relative index.
  */
  void relExtinct(IndexT nodeIdx,
		  IndexT stIdx);

  
  /**
     @brief Accessor for 'stPath' field.
   */
  class IdxPath* getSubtreePath() const {
    return rootPath.get();
  }


  /**
     @brief Looks up the number of splitable nodes in a previously-split
     layer.

     @param del is the number of layers back to look.

     @return count of splitable nodes at layer of interest.
  */
  IndexT getSplitCount(unsigned int del) const;

  
  /**
   @brief Flushes MRRA for a pair and instantiates definition at front layer.

   @param spliCoord is the layer-relative coordinate.
 */
  void reachFlush(const SplitCoord& splitCoord);


  /**
     @brief Locates index of ancestor several layers back.

     @param reachLayer is the reaching layer.

     @param splitIdx is the index of the node reached.

     @return layer-relative index of ancestor node.
 */
  IndexT getHistory(const class DefFrontier* reachLayer,
		    IndexT splitIdx) const;


  SplitCoord getHistory(const class DefFrontier* reachLayer,
			 const SplitCoord& coord) const;
  
  
  /**
     @brief Looks up the layer containing the MRRA of a pair.
   */
  inline class DefFrontier* reachLayer(const SplitCoord& coord) const {
    return layer[layerDelta[coord.strideOffset(nPred)]].get();
  }


  /**
     @brief Accessof for splitable node count in front layer.

     @return split count.
   */
  inline IndexT getSplitCount() const {
    return splitCount;
  }


  void setStageCount(const SplitCoord& splitCoord,
		     IndexT idxImplicit,
		     IndexT rankCount);

  
  void setStageCount(const SplitCoord& splitCoord,
		     const StageCount& sc) const;

};


#endif

