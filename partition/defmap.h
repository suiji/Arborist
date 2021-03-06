// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file defmap.h

   @brief Manages the lazy repartitioning of the observation set.

   Splitting requires accessing the observations in sorted or grouped
   form.  Algorithms that do not attempt to split every node/predictor
   pair, such as Random Forest, can improve training speed by restaging
   (repartitioning) lazily.

   @author Mark Seligman
 */

#ifndef PARTITION_DEFMAP_H
#define PARTITION_DEFMAP_H


#include <deque>
#include <vector>
#include <map>

#include "splitcoord.h"
#include "typeparam.h"


/**
   @brief Manages definitions reaching the frontier.
 */
class DefMap {
  const class TrainFrame* frame;
  const PredictorT nPred; // Number of predictors.
  const PredictorT nPredFac; // Number of factor-valued predictors.

  static constexpr double efficiency = 0.15; // Work efficiency threshold.

  unique_ptr<class IdxPath> stPath; // IdxPath accessed by subtree.
  IndexT splitPrev; // # nodes in previous layer.
  IndexT splitCount; // # nodes in the layer about to split.
  const class Layout* layout;
  const IndexT noRank;
  const PredictorT nPredDense; // Number of predictors using dense indexing.
  const vector<IndexT> denseIdx; // # Compressed mapping to dense offsets.
  vector<class PreCand> restageCand;
  unique_ptr<class ObsPart> obsPart;

  vector<unsigned int> history; // Current layer's history.
  vector<unsigned int> historyPrev; // Previous layer's history:  accum.
  vector<unsigned char> layerDelta; // # layers back split was defined.
  vector<unsigned char> deltaPrev; // Previous layer's delta:  accum.
  deque<unique_ptr<class DefLayer> > layer; // Caches layers tracked by history.
  vector<PredictorT> runCount;
  

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
     @brief Adds new definitions for all predictors at the root layer.
  */
  void rootDef(PredictorT predIdx,
	       bool singleton,
	       IndexT implicitCount);


  /**
     @brief Class constructor.

     @param bagCount enables sizing of predicate bit vectors.

     @param splitCount specifies the number of splits to map.
  */
  DefMap(const class TrainFrame* frame,
         unsigned int bagCount);

  /**
     @brief Class finalizer.
  */
  ~DefMap();

 /**
     @brief Pushes first layer's path maps back to all back layers
     employing node-relative indexing.
  */
  void backdate() const;


  /**
     @brief Dense offsets maintained separately, as a special case.

     @return offset strided by 'nPredDense'.
   */
  inline IndexT denseOffset(const SplitCoord& splitCoord) const {
    return splitCoord.nodeIdx * nPredDense + denseIdx[splitCoord.predIdx];
  }


  inline IndexT denseOffset(const PreCand& cand) const {
    return denseOffset(cand.splitCoord);
  }


  PredictorT getNPredDense() const {
    return nPredDense;
  }


  class DefLayer* getLayer(unsigned int del) const {
    return layer[del].get();
  }


  /**
     @brief Delayed erasure of rear layers.

     Reaching layers must persist through restaging ut allow path lookup.
     @param flushCount is the number of rear layers to erase.
  */
  void eraseLayers(unsigned int flushCount);
  
  /**
     @brief Flushes reaching definition and preschedules.
  */
  unsigned int preschedule(const SplitCoord& splitCoord,
			   vector<PreCand>& preCand);
  
  /**
     @brief Passes through to front layer.
   */
  bool isSingleton(const PreCand& defCoord) const;

  
  bool isSingleton(const PreCand& defCoord,
		   PredictorT& runCount) const;
  
  
  /**
     @brief Passes through to front layer.
   */
  void adjustRange(const PreCand& preCand,
		   IndexRange& idxRange) const;


  IndexT* getBufferIndex(const class SplitNux* nux) const;

  
  class SampleRank* getPredBase(const SplitNux* nux) const;

  
  /**
     @brief Passes through to front layer.
   */
  IndexT getImplicitCount(const PreCand& preCand) const;


  /**
     @brief Passes ObsPart through to Sample method.
   */
  void stage(const class Sample* sample);


  void branchUpdate(const class SplitNux* nux,
		    const vector<IndexRange>& range,
		    class BranchSense* branchSense,
		    struct CritEncoding& enc) const;


  void branchUpdate(const class SplitNux* nux,
		    const IndexRange& range,
		    class BranchSense* branchSense,
		    struct CritEncoding& enc) const;

  /**
     @brief Appends a restaging candidate.
   */
  void restageAppend(const class PreCand& cand);

  /**
     @brief Updates the data (observation) partition.
   */
  vector<class PreCand> restage(class SplitFrontier* splitFrontier);



  /**
     @brief Repartitions observations at a specified cell.

     @param mrra contains the coordinates of the originating cell.
   */
  void restage(const PreCand& mrra) const;
  
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

     @param predIdx is the predictor index.

     @param nStride is the stride multiple.

     @param[out] facStride is the strided factor index for dense lookup.

     @return true iff predictor is factor-valude.
   */
  bool factorStride(const SplitCoord& splitCoord,
                    unsigned int& facStride) const;


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
     @return 'noRank' value for the current subtree.
   */
  inline unsigned int getNoRank() const {
    return noRank;
  }



  /**
     @brief Looks up the number of splitable nodes in a previously-split
     layer.

     @param del is the number of layers back to look.

     @return count of splitable nodes at layer of interest.
  */
  unsigned int getSplitCount(unsigned int del) const;

  
  /**
     @brief Flips source bit if a definition reaches to current layer.
  */
  void addDef(const PreCand& splitCoord,
              bool singleton);

  /**
     @brief Sets pair as singleton at the front layer.

     @param splitIdx is the layer-relative node index.

     @param predIdx is the predictor index.
  */
  void setSingleton(const SplitCoord& splitCoord) const;


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


  
  /**
     @brief Numeric run counts are constrained to be either 1, if singleton,
     or zero otherwise.

     Singleton iff (dense and all indices implicit) or (not dense and all
     indices have identical rank).
  */
  inline void setRunCount(const SplitCoord& splitCoord,
                          bool hasImplicit,
                          PredictorT rankCount) {
    PredictorT rCount = rankCount + (hasImplicit ? 1 : 0);
    if (rCount == 1) {
      setSingleton(splitCoord);
    }

    IndexT facStride;
    if (factorStride(splitCoord, facStride)) {
      runCount[facStride] = rCount;
    }
  }

  
  /**
     @brief Determines run count currently associated with a split coordinate.
   */
  inline PredictorT getRunCount(const PreCand& defCoord) const {
    IndexT facStride;
    return factorStride(defCoord.splitCoord, facStride) ? runCount[facStride] : 0;
  }
};


#endif

