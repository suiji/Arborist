// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file deffrontier.h

   @brief Manages the lazy repartitioning of the observation set.

   Splitting requires accessing the observations in sorted/grouped
   form.  Algorithms that do not attempt to split every node/predictor
   pair, such as Random Forest, can improve training speed by performing
   this updating (repartitioning) lazily.

   @author Mark Seligman
 */

#ifndef PARTITION_DEFFRONTIER_H
#define PARTITION_DEFFRONTIER_H


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

  /**
     @brief MRRA component initialized at construction, StageCount at (re)staging.
   */
  PreCand(const SplitCoord& splitCoord,
	  unsigned int bufIdx) :
    mrra(MRRA(splitCoord, bufIdx, 0)) {
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
class DefFrontier {
  const PredictorT nPred; // Number of predictors.
  const class Frontier* frontier;
  const IndexT bagCount;
  
  static constexpr double efficiency = 0.15; // Work efficiency threshold.

  unique_ptr<class IdxPath> stPath; // IdxPath accessed by subtree.
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
  deque<unique_ptr<class DefLayer> > layer; // Caches layers tracked by history.
  vector<vector<PreCand>> preCand; // Restageable, possibly splitable, coordinates.
  //  vector<PredictorT> runCount;
  

  /**
     @brief Increments reaching layers for all pairs involving node.

     @param splitIdx is the index of a splitting node w.r.t. current layer.

     @param parIdx is the index of the parent w.r.t. previous layer.
   */
  inline void inherit(unsigned int splitIdx, unsigned int parIdx) {
    unsigned char *colCur = &layerDelta[splitIdx * nPred];
    unsigned char *colPrev = &deltaPrev[parIdx * nPred];
    for (unsigned int predIdx = 0; predIdx < nPred; predIdx++) {
      colCur[predIdx] = colPrev[predIdx] + 1;
    }
  }


 public:

  /**
     @brief Class constructor.

     @param frame_ is the training frame.

     @param frontier_ tracks the frontier nodes.
  */
  DefFrontier(const class TrainFrame* frame,
	      const class Frontier* frontier);

  
  /**
     @brief Class finalizer.
  */
  ~DefFrontier();

 /**
     @brief Pushes first layer's path maps back to all back layers
     employing node-relative indexing.
  */
  void backdate() const;


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


  class DefLayer* getLayer(unsigned int del) const {
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
  void initPrecand();

  
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
     @brief Flushes reaching definition and reports schulability.
     
     @param[out] outputs the cell's buffer index, if splitable.

     @return true iff cell at coordinate is splitable.
  */
  bool preschedule(const SplitCoord& splitCoord,
		   unsigned int& bufIdx);


  /**
     @brief As above, but discovers buffer via lookup.
   */
  bool preschedule(const SplitCoord& splitCoord);

  
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


  IndexT* getBufferIndex(const class SplitNux* nux) const;

  
  class SampleRank* getPredBase(const SplitNux* nux) const;

  
  /**
     @brief Passes through to front layer.
   */
  IndexT getImplicitCount(const MRRA& preCand) const;


  /**
     @brief Passes ObsPart through to Sample method.
   */
  void stage(const class Sample* sample);


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

     @param splitNext is the number of splitable nodes in the current
     subtree layer.

     @param idxLive is the number of live indices.

     @param nodeRel is true iff the indexing regime is node-relative.
  */
  void overlap(unsigned int splitNext,
               unsigned int bagCount,
               unsigned int idxLive,
               bool nodeRel);


  /**
     @brief Consumes all fields from an IndexSet relevant to restaging.

     @param layerIdx is the layer-relative index of the successor node.

     @param par is the index of the splitting parent.

     @param start is the cell starting index.

     @param extent is the index count.

     @param relBase

     @param path is a unique path identifier.
  */
  void reachingPath(IndexT layerIdx,
                    IndexT parIdx,
                    const IndexRange& bufRange,
                    IndexT relBase,
                    unsigned int path);
  
  /**
     @brief Flushes non-reaching definitions as well as those about
     to fall off the layer deque.

     @return count of layers to flush.
  */
  unsigned int flushRear();


  /**
     @brief Pass-through for strided factor offset.

     @param splitCoord is the node/predictor pair.

     @param[out] facStride is the strided factor index for dense lookup.

     @return true iff predictor is factor-valued.
   */
  //  bool isFactor(const SplitCoord& splitCoord,
  //		unsigned int& facStride) const; // EXIT


  /**
     @brief Updates both node-relative path for a live index, as
     well as subtree-relative if back layers warrant.

     @param ndx is a node-relative index from the previous layer.

     @param targIdx is the updated node-relative index:  current layer.

     @param stx is the associated subtree-relative index.

     @param path is the path reaching the target node.

     @param ndBase is the base index of the target node:  current layer.
   */
  void setLive(unsigned int ndx,
               unsigned int targIdx,
               unsigned int stx,
               unsigned int path,
               unsigned int ndBase);


  /**
     @brief Marks subtree-relative path as extinct, as required by back layers.

     @param stIdx is the subtree-relatlive index.
  */
  void setExtinct(IndexT stIdx);


  /**
     @brief Terminates node-relative path an extinct index.  Also
     terminates subtree-relative path if currently live.

     @param nodeIdx is a node-relative index.

     @param stIdx is the subtree-relative index.
  */
  void setExtinct(unsigned int nodeIdx, IndexT stIdx);

  
  /**
     @brief Accessor for 'stPath' field.
   */
  class IdxPath *getSubtreePath() const {
    return stPath.get();
  }


  /**
     @brief Looks up the number of splitable nodes in a previously-split
     layer.

     @param del is the number of layers back to look.

     @return count of splitable nodes at layer of interest.
  */
  unsigned int getSplitCount(unsigned int del) const;

  
  /**
     @brief Looks up front path belonging to a back layer.

     @param del is the number of layers back to look.

     @return back layer's front path.
  */
  const class IdxPath *getFrontPath(unsigned int del) const;

  
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
  IndexT getHistory(const DefLayer *reachLayer,
		    IndexT splitIdx) const;


  SplitCoord getHistory(const DefLayer* reachLayer,
			 const SplitCoord& coord) const;
  
  
  /**
     @brief Looks up the layer containing the MRRA of a pair.
   */
  inline class DefLayer* reachLayer(const SplitCoord& coord) const {
    return layer[layerDelta[coord.strideOffset(nPred)]].get();
  }


  /**
     @brief Accessof for splitable node count in front layer.

     @return split count.
   */
  inline unsigned int getSplitCount() const {
    return splitCount;
  }


  void setStageCount(const SplitCoord& splitCoord,
		     IndexT idxImplicit,
		     IndexT rankCount);

  
  void setStageCount(const SplitCoord& splitCoord,
		     const StageCount& sc) const;

};


#endif

