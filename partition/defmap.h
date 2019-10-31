// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file defmap.h

   @brief Definitions for the classes managing the most recently
   trained frontier layers.

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
   @brief Class managing the most recent level of the tree.
 */
class DefMap {
  const class SummaryFrame* frame;
  const unsigned int nPred; // Number of predictors.
  const unsigned int nPredFac; // Number of factor-valued predictors.

  static constexpr double efficiency = 0.15; // Work efficiency threshold.

  unique_ptr<class IdxPath> stPath; // IdxPath accessed by subtree.
  IndexT splitPrev; // # nodes in previous layer.
  IndexT splitCount; // # nodes in the layer about to split.
  const class RankedFrame *rankedFrame;
  const unsigned int noRank;

  vector<unsigned int> history; // Current layer's history.
  vector<unsigned int> historyPrev; // Previous layer's history:  accum.
  vector<unsigned char> layerDelta; // # layers back split was defined.
  vector<unsigned char> deltaPrev; // Previous layer's delta:  accum.
  deque<unique_ptr<class DefLayer> > layer; // Caches layers tracked by history.
  vector<unsigned int> runCount;
  

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

     @param stageCount is a vector of per-predictor staging statistics.
  */
  void rootDef(const vector<struct StageCount>& stageCount,
               unsigned int bagCount);


  /**
     @brief Class constructor.

     @param bagCount enables sizing of predicate bit vectors.

     @param splitCount specifies the number of splits to map.
  */
  DefMap(const class SummaryFrame* frame,
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

  

  class DefLayer* getLayer(unsigned int del) const {
    return layer[del].get();
  }


  /**
     @brief Delayed erasure of rear layers.

     Reaching layers must persist through restaging ut allow path lookup.
     @param flushCount is the number of rear layers to erase.
  */
  void
  eraseLayers(unsigned int flushCount);
  
  /**
     @brief Flushes reaching definition and preschedules.

     @return 1 iff not singleton else 0.
  */
  unsigned int 
  preschedule(class SplitFrontier* splitFrontier,
	      const SplitCoord& splitCoord,
	      vector<struct DefCoord>& preCand) const;

  
  /**
     @brief Passes through to front layer.
   */
  bool
  isSingleton(const DefCoord& defCoord) const;


  bool
  isSingleton(const SplitCoord& splitCoord,
	      DefCoord& defCoord) const;

  
  /**
     @brief Passes through to front layer.
   */
  IndexRange
  adjustRange(const struct DefCoord& preCand,
	      const class SplitFrontier* splitFrontier) const;

  /**
     @brief Passes through to front layer.
   */
  IndexT
  getImplicitCount(const struct DefCoord& preCand) const;


  /**
     @brief Repartitions observations at a specified cell.

     @param mrra contains the coordinates of the originating cell.
   */
  void
  restage(class ObsPart* obsPart,
	  const DefCoord& mrra) const;
  
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
  unsigned int flushRear(class SplitFrontier* splitFrontier);


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
  void addDef(const DefCoord& splitCoord,
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
  void reachFlush(class SplitFrontier* splitFrontier,
		  const SplitCoord& splitCoord) const;


  /**
     @brief Locates index of ancestor several layers back.

     @param reachLayer is the reaching layer.

     @param splitIdx is the index of the node reached.

     @return layer-relative index of ancestor node.
 */
  IndexT
  getHistory(const DefLayer *reachLayer,
	     IndexT splitIdx) const;


  SplitCoord
  getHistory(const DefLayer* reachLayer,
	     const SplitCoord& coord) const;
  
  
  /**
     @brief Looks up the layer containing the MRRA of a pair.
   */
  inline class DefLayer*
  reachLayer(const SplitCoord& coord) const {
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
     @brief Determines run count currently associated with a predictor.
   */
  inline PredictorT getRunCount(const DefCoord& defCoord) const {
    IndexT facStride;
    return factorStride(defCoord.splitCoord, facStride) ? runCount[facStride] : 0;
  }
};


#endif

