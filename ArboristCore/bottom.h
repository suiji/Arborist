// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file bottom.h

   @brief Definitions for the classes managing the most recently
   trained tree levels.

   @author Mark Seligman

 */

#ifndef ARBORIST_BOTTOM_H
#define ARBORIST_BOTTOM_H

#include <deque>
#include <vector>
#include <map>

#include "typeparam.h"


/**
   @brief Coordinates referencing most-recently restaged ancester (MRRA).
 */
class RestageCoord {
  SPPair mrra; // Level-relative coordinates of reaching ancestor.
  unsigned char del; // # levels back to referencing level.
  unsigned char bufIdx; // buffer index of mrra's SamplePred.
 public:

  void inline init(const SPPair &_mrra, unsigned int _del, unsigned int _bufIdx) {
    mrra = _mrra;
    del = _del;
    bufIdx = _bufIdx;
  }

  void inline Ref(SPPair &_mrra, unsigned int &_del, unsigned int &_bufIdx) {
    _mrra = mrra;
    _del = del;
    _bufIdx = bufIdx;
  }
};


/**
   @brief Class managing the most recent level of the tree.
 */
class Bottom {
  const unsigned int nPred; // Number of predictors.
  const unsigned int nPredFac; // Number of factor-valued predictors.
  const unsigned int bagCount; // Count of uniquely-sampled rows.

  static constexpr double efficiency = 0.15; // Work efficiency threshold.

  class IdxPath *stPath; // IdxPath accessed by subtree.
  unsigned int splitPrev; // # nodes in previous level.
  unsigned int splitCount; // # nodes in the level about to split.
  const class FrameTrain *frameTrain;
  const class RowRank *rowRank;
  const unsigned int noRank;

  vector<unsigned int> history; // Current level's history.
  vector<unsigned int> historyPrev; // Previous level's history:  accum.
  vector<unsigned char> levelDelta; // # levels back split was defined.
  vector<unsigned char> deltaPrev; // Previous level's delta:  accum.
  class Level *levelFront; // Current level.
  vector<unsigned int> runCount;
  deque<class Level *> level; // However may levels are tracked by history.
  
  vector<RestageCoord> restageCoord;

  /**
     @brief General, multi-level restaging.
  */
  void restage(class SamplePred *samplePred, RestageCoord &rsCoord);

  /**
     @brief Pushes first level's path maps back to all back levels
     employing node-relative indexing.
  */
  void backdate() const;

  
  /**
     @brief Increments reaching levels for all pairs involving node.

     @param splitIdx is the index of a splitting node w.r.t. current level.

     @param parIdx is the index of the parent w.r.t. previous level.
   */
  inline void inherit(unsigned int splitIdx, unsigned int parIdx) {
    unsigned char *colCur = &levelDelta[splitIdx * nPred];
    unsigned char *colPrev = &deltaPrev[parIdx * nPred];
    for (unsigned int predIdx = 0; predIdx < nPred; predIdx++) {
      colCur[predIdx] = colPrev[predIdx] + 1;
    }
  }


 public:

  /**
     @brief Adds new definitions for all predictors at the root level.

     @param stageCount is a vector of per-predictor staging statistics.
  */
  void rootDef(const vector<class StageCount>& stageCount);

  
  /**
     @brief Schedules a reaching definition for restaging.

     @param del is the number of levels back that the definition resides.

     @param mrraIdx is the level-relative index of the defining node.

     @param predIdx is the predictor index.

     @param bufIdx is the buffer in which the definition resides.
   */
  void scheduleRestage(unsigned int del,
                       unsigned int mrraIdx,
                       unsigned int predIdx,
                       unsigned int bufIdx);


  /**
     @brief Class constructor.

     @param bagCount enables sizing of predicate bit vectors.

     @param splitCount specifies the number of splits to map.
  */
  Bottom(const class FrameTrain* frameTrain_,
         const class RowRank* rowRank_,
         unsigned int bagCount_);

  /**
     @brief Class finalizer.
  */
  ~Bottom();


  /**
     @brief Entry to restaging and candidate scheduling.
  */
  void scheduleSplits(class SamplePred *samplePred,
                      class SplitNode* splitNode,
                      class IndexLevel *index);

  
  /**
     @brief Updates subtree and pretree mappings from temporaries constructed
     during the overlap.  Initializes data structures for restaging and
     splitting the current level of the subtree.

     @param splitNext is the number of splitable nodes in the current
     subtree level.

     @param idxLive is the number of live indices.

     @param nodeRel is true iff the indexing regime is node-relative.
  */
  void overlap(unsigned int splitNext,
               unsigned int idxLive,
               bool nodeRel);


  /**
     @brief Consumes all fields from an IndexSet relevant to restaging.

     @param levelIdx is the level-relative index of the successor node.

     @param par is the index of the splitting parent.

     @param start is the cell starting index.

     @param extent is the index count.

     @param relBase

     @param path is a unique path identifier.
  */
  void reachingPath(unsigned int levelIdx,
                    unsigned int parIdx,
                    unsigned int start,
                    unsigned int extent,
                    unsigned int relBase,
                    unsigned int path);
  
  /**
     @brief Flushes non-reaching definitions as well as those about
     to fall off the level deque.

     @return highest level not flushed.
  */
  unsigned int flushRear();


  /**
     @brief Restages predictors and splits as pairs with equal priority.

     @param samplePred contains the compressed observation set.
  */
  void restage(class SamplePred *samplePred);


  /**
     @brief Pass-through for strided factor offset.

     @param predIdx is the predictor index.

     @param nStride is the stride multiple.

     @param[out] facStride is the strided factor index for dense lookup.

     @return true iff predictor is factor-valude.
   */
  bool factorStride(unsigned int predIdx,
                    unsigned int nStride,
                    unsigned int &facStride) const;


  /**
     @brief Updates both node-relative path for a live index, as
     well as subtree-relative if back levels warrant.

     @param ndx is a node-relative index from the previous level.

     @param targIdx is the updated node-relative index:  current level.

     @param stx is the associated subtree-relative index.

     @param path is the path reaching the target node.

     @param ndBase is the base index of the target node:  current level.
   */
  void setLive(unsigned int ndx,
               unsigned int targIdx,
               unsigned int stx,
               unsigned int path,
               unsigned int ndBase);


  /**
     @brief Marks subtree-relative path as extinct, as required by back levels.

     @param stIdx is the subtree-relatlive index.
  */
  void setExtinct(unsigned int stIdx);


  /**
     @brief Terminates node-relative path an extinct index.  Also
     terminates subtree-relative path if currently live.

     @param nodeIdx is a node-relative index.

     @param stIdx is the subtree-relative index.
  */
  void setExtinct(unsigned int nodeIdx, unsigned int stIdx);

  
  /**
     @brief Accessor for 'stPath' field.
   */
  class IdxPath *subtreePath() const {
    return stPath;
  }
  

  /**
     @return 'noRank' value for the current subtree.
   */
  inline unsigned int getNoRank() const {
    return noRank;
  }



  /**
     @brief Looks up the number of splitable nodes in a previously-split
     level.

     @param del is the number of levels back to look.

     @return count of splitable nodes at level of interest.
  */
  unsigned int getSplitCount(unsigned int del) const;

  
  /**
     @brief Flips source bit if a definition reaches to current level.
  */
  void addDef(unsigned int reachIdx,
              unsigned int predIdx,
              unsigned int bufIdx,
              bool singleton);

  
  /**
     @brief Determines whether a pair references a singleton.

     @param levelIdx is the level-relative node index.

     @param predIdx is the predictor index.

     @return true iff the pair is a singleton.
   */
  bool isSingleton(unsigned int levelIdx, unsigned int predIdx) const;


  /**
     @brief Sets pair as singleton at the front level.

     @param splitIdx is the level-relative node index.

     @param predIdx is the predictor index.
  */
  void setSingleton(unsigned int levelIdx, unsigned int predIdx) const;


  /**
     @brief Invokes dense-value adjustment from front level.

     @return 
  */
  unsigned int adjustDense(unsigned int levelIdx,
                           unsigned int predIdx,
                           unsigned int &startIdx,
                           unsigned int &extent) const;

  
  /**
     @brief Looks up front path belonging to a back level.

     @param del is the number of levels back to look.

     @return back level's front path.
  */
  const class IdxPath *getFrontPath(unsigned int del) const;

  
  /**
   @brief Flushes MRRA for a pair and instantiates definition at front level.

   @param splitIdx is the level-relative node index.

   @param predIdx is the predictor index.
 */
  void reachFlush(unsigned int splitIdx, unsigned int predIdx) const;


  /**
     @brief Locates index of ancestor several levels back.

     @param reachLevel is the reaching level.

     @param splitIdx is the index of the node reached.

     @return level-relative index of ancestor node.
 */
  unsigned int getHistory(const Level *reachLevel,
                          unsigned int splitIdx) const;

  
  /**
     @brief Looks up the level containing the MRRA of a pair.
   */
  inline class Level *reachLevel(unsigned int levelIdx,
                                 unsigned int predIdx) const {
    return level[levelDelta[levelIdx * nPred + predIdx]];
  }


  /**
     @brief Accessof for splitable node count in front level.

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
  inline void setRunCount(unsigned int splitIdx,
                          unsigned int predIdx,
                          bool hasImplicit,
                          unsigned int rankCount) {
    unsigned int rCount = rankCount + (hasImplicit ? 1 : 0);
    if (rCount == 1) {
      setSingleton(splitIdx, predIdx);
    }

    unsigned int facStride;
    if (factorStride(predIdx, splitIdx, facStride)) {
      runCount[facStride] = rCount;
    }
  }


  /**
     @brief Looks up the run count associated with a given node, predictor pair.
     
     @param splitIdx is the level-relative node index.

     @param predIdx is the predictor index.

     @return run count associated with the node, if factor, otherwise zero.
   */
  inline unsigned int getRunCount(unsigned int splitIdx,
                               unsigned int predIdx) const {
    unsigned int facStride;
    return factorStride(predIdx, splitIdx, facStride) ? runCount[facStride] : 0;
  }
};


#endif

