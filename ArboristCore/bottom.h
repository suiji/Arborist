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

/**
   @brief Records sample's recent branching path.
 */
class SamplePath {
  unsigned char extinct; // Sticky semantics.
  unsigned char path;
 public:

  SamplePath();

  inline void PathLeft() {
    path = (path << 1) | 0;
  }


  inline void PathRight() {
    path = (path << 1) | 1;
  }


  inline void PathExtinct() {
    extinct = 1;
  }
  
  
  /**
     @brief Accessor.

     @param _path outputs the path reaching the sample, if live.  Otherwise
     the value is undefined.

     @return whether sample is live.
   */
  inline bool IsLive(unsigned int &_path) const {
    _path = path;
    return extinct == 0;
  }


  inline int Path(unsigned int del) const {
    return extinct == 0 ? path & ~(0xff << del) : -1;
  }
};


/**
   @brief Records node and offset reached by path from MRRA.
 */
class PathNode {
  int levelIdx; // Negative iff path extinct.
  int offset; // Target offset for path.
 public:

  /**
     @brief Initializes to extinct path.
   */
  void Init() {
    levelIdx = -1;
    offset = -1;
  }

  
  /**
     @brief Sets to non-extinct path coordinates.
   */
  inline void Init(unsigned int _levelIdx, unsigned int _offset) {
    levelIdx = _levelIdx;
    offset = _offset;
  }
  

  inline void Coords(int &_levelIdx, int &_offset) const {
    _offset = offset;
    _levelIdx = levelIdx;
  }

  
  inline int Offset() const {
    return offset;
  }
};


class RestageNode {
  unsigned int startIdx;
  unsigned int extent;// Requires access to start, end
  unsigned int pathZero; // Beginning index of path offsets within 'pathAccum'.
  unsigned char levelDel; // Level difference between creation and restaging.
  void Singletons(class Bottom *bottom, const std::vector<PathNode> &pathNode, const int targOffset[], const class SPNode targ[], unsigned int predIdx) const;
 public:

  void Restage(class Bottom *bottom, class SamplePred *samplePred, const std::vector<PathNode> &pathNode, unsigned int predIdx, unsigned int sourceBit) const;
  void RestageTwo(class Bottom *bottom, class SamplePred *samplePred, const std::vector<PathNode> &pathNode, unsigned int predIdx, unsigned int sourceBit) const;


  /**
     @brief Initializes the node.  The first three parameters are the immutable state of the MRRA.

     @param _pathZero is an accumulated starting index for restaging targets.

     @return void.
   */
  inline void Init(unsigned int _startIdx, unsigned int _extent, unsigned int _levelDel, unsigned int _pathZero) {
    startIdx = _startIdx;
    extent = _extent;
    pathZero = _pathZero;
    levelDel = _levelDel;
  }


  /**
     @brief Computes the offset for a target path.
   */
  inline unsigned int PathOffset(unsigned int path, unsigned int &_levelDel) const {
    _levelDel = levelDel;
    return pathZero + (path & ~(0xff << levelDel));
  }
};


/**
   @brief Bundling of RestageNode and predictor indices:  1/many.
 */
class RestagePair {
  unsigned int nodeIdx; // RestageNode index.
  unsigned int predIdx; // Predictor index.
 public:

  inline void Init(unsigned int _nodeIdx, unsigned int _predIdx) {
    nodeIdx = _nodeIdx;
    predIdx = _predIdx;
  }
  

  void Coords(int &_nodeIdx, int &_predIdx) const {
    _nodeIdx = nodeIdx;
    _predIdx = predIdx;
  }
};


/**
  @brief Most-recently restaged ancestor.
*/

class MRRA {
  int restageIdx; // Index for external reference.  Cached for level.
  unsigned int start; // Starting index of cell within buffer.
  unsigned int extent; // Count of indices within cell.

public:
  void Init(unsigned int _start, unsigned int _extent) {
    restageIdx = -1;
    start = _start;
    extent = _extent;
  }


  /**
     @brief Accessor.
   */
  unsigned int RestageIdx() {
    return restageIdx;
  }


  void Reset() { // Must be called for reuse in subsequent levels.
    restageIdx = -1;
  }

  
  unsigned int PathAccum(unsigned int levelDel, unsigned int &pathAccum, std::vector<RestageNode> &restageNode);
};


/**
   @brief Guides splitting and memory-locality operations for the most recently
   trained tree levels.
 */
class BottomNode {
  unsigned int runCount;
  unsigned int mrraIdx; // Level-relative node position from which restaging.
  unsigned char levelDel; // # levels back at which restaging occurred:  <= pathMax.

 public:
  static constexpr unsigned int pathMax = 8 * sizeof(unsigned char);

  
  /**
     @brief Level-zero initializer.

     @param _runCount is an upper bound on the number of factor levels.

     @return void.
   */
  inline void Init(unsigned int _runCount) {
    runCount = _runCount;
    mrraIdx = 0;
    levelDel = 0;
  }

  
  /**
     @brief Sets or Inherits values derived from parent.

     @return void.
   */
  inline void Inherit(BottomNode &botLevel) {
    runCount = botLevel.runCount;
    mrraIdx = botLevel.mrraIdx;
    levelDel = botLevel.levelDel + 1;
  }

  
  /**
     @brief Indicates whether the node's MRRA requires requres restaging.

     @return true iff distance to MRRA's level has reached the path maximum.
   */
  inline bool Exhausted() const {
    return levelDel >= pathMax;
  }


  /**
     @brief Accessor for runCount field.
   */  
  inline unsigned int RunCount() const {
    return runCount;
  }


  /**
     @brief Setter for runCount.
   */
  inline void RunCount(int _runCount) {
    runCount = _runCount;
  }


  /**
     @brief Resets MRRA index and level delta to ajdust for effects of
     restaging.  Crucial for inheritance at next level. 
   */
  inline void MrraReset(unsigned int _levelIdx) {
    mrraIdx = _levelIdx;
    levelDel = 0;
  }

  
  /**
     @brief Accessor for MRRA, which can be reset.

     @param levelIdx is the level-relative node position.

     @param _levelDel outputs level delta.

     @param reset indicates whether to reset index and delta (for inheritance).

     @return MRRA index and reference to level delta.
   */
  inline unsigned int MrraIdx(unsigned int levelIdx, unsigned int &_levelDel) {
    _levelDel = levelDel;
    return mrraIdx;
  }


  /**
     @brief Destructive accessor for MRRA index.

     @param levelIdx is the level-relative node position.

     @return MRRA index.
   */
  inline unsigned int MrraConsume(unsigned int levelIdx) {
    unsigned int _mrraIdx = mrraIdx;
    MrraReset(levelIdx);

    return _mrraIdx;
  }
};


class SplitPair {
  static const int noSplit = -2;
  unsigned int bottomIdx;
  int restageIdx; // Dense numbering of MRRAs reaching this level.
  int setIdx;

 public:
  
  inline void Init(unsigned int _bottomIdx, int _restageIdx) {
    bottomIdx = _bottomIdx;
    restageIdx = _restageIdx;
    setIdx = noSplit;
  }

  
  inline void SplitInit(unsigned int _bottomIdx, int _restageIdx, int _setIdx = -1) {
    bottomIdx = _bottomIdx;
    restageIdx = _restageIdx;
    setIdx = _setIdx;
  }


  /**
     @brief Accessor for 'bottomIdx'.
   */
  inline unsigned int BottomIdx() const {
    return bottomIdx;
  }


  inline unsigned int BottomIdx(unsigned int &_restageIdx) const {
    _restageIdx = restageIdx;

    return bottomIdx;
  }

  inline int SetIdx() const {
    return setIdx;
  }
};


class Bottom {
  std::deque<class BitMatrix *> bufferLevel;
  std::deque<std::vector<MRRA> > mrraLevel;
  std::vector<BottomNode> bottomNode; // All levelCount x nPred cells referenceable at current level.
  std::vector<BottomNode> preStage; // Temporary staging area.


  SamplePath *samplePath;
  const unsigned int nPred;
  const unsigned int nPredFac;
  unsigned int ancTot; // Current count of extant ancestors.
  unsigned int levelCount; // # nodes in the level about to split.
  class SamplePred *samplePred;
  class SplitPred *splitPred;  // constant?
  class SplitSig *splitSig;
  class Run *run; // Passes through to SSNode methods.
  
  //unsigned int rhIdxNext; // GPU client only:  Starting RHS index.

  int RestageIdx(unsigned int bottomIdx);
  class BitMatrix *RestageSetup(const bool splitFlags[], std::vector<RestageNode> &restageNode, std::vector<PathNode> &pathNode, std::vector<RestagePair> &restagePair);
  void PairInit(const bool splitFlags[], const BitMatrix *ancReach, const class IndexNode indexNode[], std::vector<RestageNode> &restageNode, std::vector<PathNode> &pathNode, std::vector<RestagePair> &restagePair);
  void SourceTarg(unsigned int levelDel, class BV *restageSource, BitMatrix *restageTarg, unsigned int restgeIdx, unsigned int botIdx, unsigned int levelIdx, unsigned int predIdx);
  void Restage(const std::vector<RestageNode> &restageNode, const std::vector<RestagePair> &restagePair, const std::vector<PathNode> &pathNode, const class BV *bufSource);
  void Split(const std::vector<SplitPair> &pairNode, const class IndexNode indexNode[]);
  void Split(const class IndexNode indexNode[], unsigned int bottomIdx, int setIdx);

  
  inline bool Singleton(unsigned int botIdx) {
    return bottomNode[botIdx].RunCount() == 1;
  }

  
  inline bool Singleton(unsigned int botIdx, int &runCount) {
    return (runCount = bottomNode[botIdx].RunCount()) == 1;
  }
  
  /**
     @brief Determines whether MRRA should be scheduled for restaging.

     @return true iff MRRA to be restaged.
   */  
  inline bool ScheduleMRRA(const bool splitFlags[], unsigned int botIdx) {
    return !Singleton(botIdx) && (splitFlags[botIdx] || Exhausted(botIdx));
  }

  
 public:
  static Bottom *FactoryReg(class SamplePred *_samplePred, unsigned int bagCount);
  static Bottom *FactoryCtg(class SamplePred *_samplePred, class SampleNode *_sampleCtg, unsigned int bagCount);
  
  Bottom(class SamplePred *_samplePred, class SplitPred *_splitPred, unsigned int bagCount, unsigned int _nPred, unsigned int _nPredFac);
  ~Bottom();
  void LevelInit();
  void Level(const bool splitFlags[], const class IndexNode indexNode[]);
  void Overlap(unsigned int _splitNext);
  void DeOverlap();
  void LevelClear();
  const std::vector<class SSNode*> LevelSplit(class Index *index, class IndexNode indexNode[]);
  void Inherit(unsigned int _splitIdx, unsigned int nodeNext);
  unsigned int PathAccum(std::vector<RestageNode> &restageNode, unsigned int bottomIdx, unsigned int &_pathAccum);
  void SSWrite(unsigned int bottomIdx, int setIdx, unsigned int lhSampCount, unsigned lhIdxCount, double info);
  unsigned int BufBit(unsigned int levelIdx, unsigned int predIdx);

  
  /**
     @brief Accessor.  SSNode only client.
   */
  inline class Run *Runs() {
    return run;
  }


  /**
     @brief Setter methods for sample path.
   */
  inline bool IsLive(unsigned int sIdx, unsigned int &sIdxPath) const {
    return samplePath[sIdx].IsLive(sIdxPath);
  }


  inline int Path(unsigned int sIdx, unsigned int del) const {
    return samplePath[sIdx].Path(del);
  }


  inline void PathLeft(unsigned int sIdx) const {
    samplePath[sIdx].PathLeft();
  }


  inline void PathRight(unsigned int sIdx) const {
    samplePath[sIdx].PathRight();
  }


  inline void PathExtinct(unsigned int sIdx) const {
    samplePath[sIdx].PathExtinct();
  }
  

  /**
     @brief Derives pair coordinates from positional index.

     @param bottomIdx is the position within a vector assumed to be in level-major order.

     @param levelIdx is the level-relative node index.

     @param predIdx is the predictor index.

     @return void, with reference paramters.
   */
  inline void SplitCoords(unsigned int bottomIdx, unsigned int &levelIdx, unsigned int &predIdx) const {
    levelIdx = bottomIdx / nPred;
    predIdx = bottomIdx - nPred * levelIdx;
  }


  /**
     @brief Derives offset from <major, minor> pair with 'nPred' as implicit stride.

     @param major is the major dimension.

     @param minor is the fastest-moving index.

     @return computed offset for pair.
   */
  inline unsigned int PairOffset(unsigned int major, unsigned int minor) const {
    return major * nPred + minor;
  }

  
  inline bool Exhausted(unsigned int idx) const {
    return bottomNode[idx].Exhausted();
  }


  inline void SetSingleton(unsigned int levelIdx, unsigned int predIdx) {
    bottomNode[levelIdx * nPred + predIdx].RunCount(1);
  }

  
  inline void RunCount(unsigned int bottomIdx, int runCount) {
    bottomNode[bottomIdx].RunCount(runCount);
  }


  inline unsigned int MrraIdx(unsigned int bottomIdx, unsigned int levelIdx, unsigned int &levelDel) {
    return bottomNode[bottomIdx].MrraIdx(levelIdx, levelDel);
  }


  inline unsigned int MrraConsume(unsigned int bottomIdx, unsigned int levelIdx) {
    return bottomNode[bottomIdx].MrraConsume(levelIdx);
  }

  
  inline MRRA &Mrra(unsigned int bottomIdx, unsigned int &levelDel) {
    unsigned int levelIdx, predIdx;
    SplitCoords(bottomIdx, levelIdx, predIdx);

    unsigned int mrraIdx = bottomNode[bottomIdx].MrraIdx(levelIdx, levelDel);
    std::vector<MRRA> &mrraVec = *(end(mrraLevel) - levelDel);
    return mrraVec[mrraIdx];
  }
};

#endif

