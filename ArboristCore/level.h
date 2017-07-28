// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file level.h

   @brief Definitions for the classes managing a single tree level.

   @author Mark Seligman

 */

#ifndef ARBORIST_LEVEL_H
#define ARBORIST_LEVEL_H

#include <vector>
#include "param.h"


/**
   @brief Coordinates from ancestor IndexSet.
 */
class IndexAnc {
  unsigned int start;
  unsigned int extent;
 public:

  inline void Init(unsigned int _start, unsigned int _extent) {
    start = _start;
    extent = _extent;
  }

  
  /**
     @brief Dual field accessor, specific to sample indexing.
   */
  inline void Ref(unsigned int &_start, unsigned int &_extent) {
    _start = start;
    _extent = extent;
  }
};


/**
   @brief Inherited state for most-recently-restaged ancestor.
 */
class MRRA {
  static const unsigned int defBit = 1;
  static const unsigned int oneBit = 2;
  static const unsigned int denseBit = 4;

  // Addition bits available for multiple buffers:
  static const unsigned int bufBit = 8;

  unsigned char raw;
 public:

 
  inline void Init() {
    raw = 0;
  }

  
  inline void Init(unsigned int bufIdx, bool singleton) {
    raw = defBit | (singleton ? oneBit : 0) | (bufIdx == 0 ? 0 : bufBit);
  }


  inline bool Singleton() const {
    return (raw & oneBit) != 0;
  }

  inline bool Singleton(unsigned int &bufIdx) const {
    bufIdx = (raw & bufBit) == 0 ? 0 : 1;
    return Singleton();
  }
  

  inline void SetDense() {
    raw |= denseBit;
  }

  
  /**
     @brief Determines whether cell requires dense placement, i.e, is either
     unaligned within a dense region or is itself dense.

     @return true iff dense bit set.
   */
  inline bool Dense() const {
    return (raw & denseBit) != 0;
  }


  inline void SetSingleton() {
    raw |= oneBit;
  }

  
  inline bool Defined() const {
    return (raw & defBit) != 0;
  }
  

  inline bool Undefine() {
    bool wasDefined = Defined();
    raw &= ~defBit;
    return wasDefined;
  }


  /**
     @brief Looks up position parameters and resets definition bit.

     @return void, with output reference parameters.
  */
  inline void Consume(unsigned int &bufIdx, bool &singleton) {
    singleton = Singleton(bufIdx);
    (void) Undefine();
  }
};


/**
   @brief Defines the parameters needed to place a dense cell with respect
   the position of its defining node.  Parameters are maintained as relative
   values to facilitate recognition of cells no longer requiring dense
   representation.
 */
class DenseCoord {
  unsigned int margin;
  unsigned int implicit; // Nonincreasing.

 public:

  /**
     @brief Applies dense parameters to offsets derived from index node.

     @param startIdx inputs the nodewise starting offset and outputs the
     same value, minus the margin.

     @param extent inputs the nodewise index count and outputs the same
     value, minus the number of implicit indices.

     @return dense count.
   */
  inline unsigned int AdjustDense(unsigned int &startIdx, unsigned int &extent) const {
    startIdx -= margin;
    extent -= implicit;
    return implicit;
  }


  /**
     @brief Sets the dense placement parameters for a cell.

     @return void.
   */
  inline void Init(unsigned int _implicit, unsigned int _margin = 0) {
    implicit = _implicit;
    margin = _margin;
  }

};


/**
   @brief Per-level reaching definitions.
 */
class Level {
  const unsigned int nPred;
  const std::vector<unsigned int> &denseIdx;
  const unsigned int nPredDense;
  const unsigned int nSplit;
  const unsigned int noIndex; // Inattainable node index value.
  const unsigned int idxLive; // Total # sample indices at level.
  const bool nodeRel;  // Subtree- or node-relative indexing.

  class Bottom *bottom;
  class SamplePred *samplePred;
  unsigned int defCount; // # live definitions.
  unsigned char del; // Position in deque.  Increments.

  // Immutable:
  //
  static unsigned int predFixed;
  static const double *predProb;

  // Persistent:
  std::vector<IndexAnc> indexAnc; // Stage coordinates, by node.

  // More elegant and parsimonious to use std::map from pair to node,
  // but hashing much too slow.
  std::vector<MRRA> def; // Indexed by pair-offset.
  std::vector<DenseCoord> denseCoord;

  // Recomputed:
  class IdxPath *relPath;
  std::vector<unsigned int> offCand;
  std::vector<class NodePath> nodePath; // Indexed by <node, predictor> pair.
  std::vector<unsigned int> liveCount; // Indexed by node.

  // Coproc staging only.
  unsigned int spanCand; // Total candidate span.
  bool Preschedule(class SplitPred *splitPred, unsigned int levelIdx, unsigned int predIdx, unsigned int extent, unsigned int &spanCand);
  void IndexRestage(const SPPair &mrra, const Level *levelFront, unsigned int bufIdx, const unsigned int reachBase[], unsigned int reachOffset[], unsigned int splitOffset[]);


 public:
  Level(unsigned int _nSplit, unsigned int _nPred, const std::vector<unsigned int> &_denseIdx, unsigned int _nPredDense, unsigned int _noIndex, unsigned int _idxLive, bool _nodeRel, class Bottom *bottom, class SamplePred *_samplePred);
  ~Level();

  static void Immutables(unsigned int _predFixed, const double _predProb[]);
  static void DeImmutables();
  void Candidates(const class IndexLevel *index, class SplitPred *splitPred);
  void CandidateProb(class SplitPred *splitPred, unsigned int splitIdx, const double ruPred[], unsigned int extent, unsigned int &offCand);
  void CandidateFixed(class SplitPred *splitPred, unsigned int splitIdx, const double ruPred[], class BHPair heap[], unsigned int extent, unsigned int &offCand);
  void Restage(SPPair &mrra, Level *levelFront, unsigned int bufIdx);
  void Restage(const SPPair &mrra, Level *levelFront, unsigned int bufIdx, const unsigned int reachBase[], unsigned int reachOffset[]);

  // COPROCESSOR:
  void IndexRestage(SPPair &mrra, const Level *levelFront, unsigned int bufIdx);
		    
  void Flush(bool forward = true);
  void FlushDef(unsigned int mrraIdx, unsigned int predIdx);
  bool NonreachPurge();
  void Paths();
  void PathInit(const class Bottom *bottom, unsigned int levelIdx, unsigned int path, unsigned int start, unsigned int extent, unsigned int relBase);
  bool ScheduleSplit(unsigned int levelIdx, unsigned int predIdx, unsigned int &rCount) const;
  void Bounds(const SPPair &mrra, unsigned int &startIdx, unsigned int &extent);
  void FrontDef(unsigned int mrraIdx, unsigned int predIdx, unsigned int bufIdx, bool singleton);
  void OffsetClone(const SPPair &mrra, unsigned int reachOffset[], unsigned int reachBase[]);
  // COPROC:
  void OffsetClone(const SPPair &mrra, unsigned int reachOffset[], unsigned int splitOffset[], unsigned int reachBase[]);
  void RunCounts(class Bottom *bottom, const SPPair &mrra, const unsigned int pathCount[], const unsigned int rankCount[]) const;

  void PackDense(unsigned int idxLeft, const unsigned int pathCount[], Level *levelFront, const SPPair &mrra, unsigned int reachOffset[]) const;
  void SetExtinct(unsigned int idx);
  bool Backdate(const class IdxPath *one2Front);
  void SetSingleton(unsigned int levelIdx, unsigned int predIdx);
  bool Splitable(unsigned int levelIdx, unsigned int predIdx, unsigned int &bufIdx);
  void SetLive(unsigned int idx, unsigned int path, unsigned int targIdx, unsigned int ndBase);


  /**
   */
  inline unsigned int Del() const {
    return del;
  }

  /**
     @brief Accessor for indexing mode.  Currently two-valued.
   */
  inline bool NodeRel() const {
    return nodeRel;
  }

  
  /**
     @brief Front path accessor.

     @return reference to front path.
   */
  inline class IdxPath *FrontPath() const {
    return relPath;
  }

  
  /**
     @brief Accessor for count of live sample indices.
  */
  inline unsigned int IdxLive() {
    return idxLive;
  }


  /**
     @brief Will overflow if level sufficiently fat.
     TODO:  switch to depth-first in such regimes.

     @return offset strided by 'nPred'.
   */
  inline unsigned int PairOffset(unsigned int mrraIdx, unsigned int predIdx) const {
    return mrraIdx * nPred + predIdx;
  }


  /**
     @brief Dense offsets maintained separately, as a special case.

     @return offset strided by 'nPredDense'.
   */
  inline unsigned int DenseOffset(unsigned int mrraIdx, unsigned int predIdx) const {
    return mrraIdx * nPredDense + denseIdx[predIdx];
  }

  
  /**
     @brief Shifts a value by the number of back-levels to compensate for
     effects of binary branching.

     @param val is the value to shift.

     @return shifted value.
   */  
  inline unsigned int BackScale(unsigned int val) const {
    return val << (unsigned int) del;
  }


  /**
     @brief Produces mask approprate for level:  lowest 'del' bits high.

     @return bit mask value.
   */
  inline unsigned int PathMask() const {
    return BackScale(1) - 1;
  }
  

  /**
     @brief Accessor.  What more can be said?

     @return definition count at this level.
  */
  inline unsigned int DefCount() {
    return defCount;
  }


  inline unsigned int SplitCount() {
    return nSplit;
  }


  /**
     @brief

     @param implicit is only set directly by staging.  Otherwise it has a
     default setting of zero, which is later reset by restaging.
   */
  inline bool Define(unsigned int levelIdx, unsigned predIdx, unsigned int bufIdx, bool singleton, unsigned int implicit = 0) {
    if (levelIdx != noIndex) {
      def[PairOffset(levelIdx, predIdx)].Init(bufIdx, singleton);
      SetDense(levelIdx, predIdx, implicit);
      defCount++;
      return true;
    }
    else {
      return false;
    }
  }


  inline void Undefine(unsigned int levelIdx, unsigned int predIdx) {
    bool wasDefined = def[PairOffset(levelIdx, predIdx)].Undefine();
    defCount -= wasDefined ? 1 : 0;
  }


  inline void Consume(unsigned int levelIdx, unsigned int predIdx, unsigned int &bufIdx, bool &singleton) {
    def[PairOffset(levelIdx, predIdx)].Consume(bufIdx, singleton);
    defCount--;
  }


  /**
     @brief Determines whether pair consists of a single run.

     @param bufIdx outputs the buffer index.

     @return true iff a singleton.
   */
  inline bool Singleton(unsigned int levelIdx, unsigned int predIdx) const {
    return def[PairOffset(levelIdx, predIdx)].Singleton();
  }


  inline bool Singleton(unsigned int levelIdx, unsigned int predIdx, unsigned int &bufIdx) const {
    return def[PairOffset(levelIdx, predIdx)].Singleton(bufIdx);
  }


  inline unsigned int AdjustDense(unsigned int levelIdx, unsigned int predIdx, unsigned int &startIdx, unsigned int &extent) const {
    return def[PairOffset(levelIdx, predIdx)].Dense() ?
      denseCoord[DenseOffset(levelIdx, predIdx)].AdjustDense(startIdx, extent) : 0;
  }


  inline void Ref(unsigned int levelIdx, unsigned int predIdx, unsigned int &bufIdx, bool &singleton) {
    singleton = def[PairOffset(levelIdx, predIdx)].Singleton(bufIdx);
  }


  inline bool Defined(unsigned int levelIdx, unsigned int predIdx) const {
    return def[PairOffset(levelIdx, predIdx)].Defined();
  }


  inline bool Dense(unsigned int levelIdx, unsigned int predIdx) const {
    return def[PairOffset(levelIdx, predIdx)].Dense();
  }

  /**
     @brief Sets the density-associated parameters for a reached node.

     @return void.
  */
  inline void SetDense(unsigned int levelIdx, unsigned int predIdx, unsigned int implicit, unsigned int margin = 0) {
    if (implicit > 0 || margin > 0) {
      def[PairOffset(levelIdx, predIdx)].SetDense();
      denseCoord[DenseOffset(levelIdx, predIdx)].Init(implicit, margin);
    }
  }


  /**
     @brief Establishes front-level IndexSet as future ancestor.

     @return void.
  */
  void Ancestor(unsigned int levelIdx, unsigned int start, unsigned int extent) {
    indexAnc[levelIdx].Init(start, extent);
  }


  void SetSpan(unsigned int _spanCand) {
    spanCand = _spanCand;
  }
};

#endif
