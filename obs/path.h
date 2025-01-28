// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file path.h

   @brief Definitions for the classes managing paths from index sets
   and to indiviual indices.

   @author Mark Seligman
 */

#ifndef PARTITION_PATH_H
#define PARTITION_PATH_H

#include <vector>

#include "typeparam.h"
#include "splitcoord.h"

/**
   @brief Records index, start and extent for path reached from MRRA.
 */
class NodePath {
  static constexpr unsigned int logPathMax = 8 * sizeof(PathT) - 1;
  // Maximal path length is also an inattainable path index.
  static constexpr unsigned int noPath = 1 << logPathMax;

  static IndexT noSplit;
  
  IndexT frontIdx; // < noIndex iff path extinct.
  IndexRange bufRange; // buffer target range for path.
  IndexT idxStart; // Node starting position in upcoming level.

public:


  NodePath() : frontIdx(noSplit),
	       bufRange(IndexRange()),
	       idxStart(0) {
  }


  /**

     @brief Sets to non-extinct path coordinates.
  */
  void init(const class IndexSet& iSet,
	    IndexT endIdx);

  
  /**
     @return maximal path length.
   */
  static constexpr unsigned int pathMax() {
    return noPath;
  }


  /**
     @brief Sets noSplit to inattainable split index.
   */
  static void setNoSplit(IndexT bagCount);

  
  /**
     @brief Determines whether a path size is representable within
     container.

     @param pathSize is the path size in question.

     @return true iff path size does not exceed maximum representable.
   */
  static bool isRepresentable(unsigned int pathSize) {
    return pathSize <= logPathMax;
  }

  
  /**
     @brief Determines whether a path is active.

     @param path is the path in in question.

     @return true iff path is active.
   */
  static bool isActive(unsigned int path) {
    return path != noPath;
  }


  /**
     @brief Multiple accessor for path coordinates.
   */
  bool getCoords(PredictorT predIdx,
			SplitCoord& coord,
			IndexRange& idxRange) const {
    if (frontIdx == noSplit) {
      return false;
    }
    else {
      idxRange = bufRange;
      coord = SplitCoord(frontIdx, predIdx);
      return true;
    }
  }

  bool getFrontIdx(IndexT& idxOut) const {
    idxOut = frontIdx;
    return (frontIdx != noSplit);
  }

  
  IndexT getIdxStart() const {
    return bufRange.getStart();
  }


  IndexT getExtent() const {
    return bufRange.getExtent();
  }
  

  IndexT getNodeStart() const {
    return idxStart;
  }


  IndexT getSplitIdx() const {
    return frontIdx;
  }
};


// Only defined for enclosing Levels employing node-relative indexing.
//
// Narrow for data locality, but wide enough to be useful.  Can
// be generalized to multiple sizes to accommodate more sophisticated
// hierarchies.
//
typedef uint_least16_t NodeRelT;

class IdxPath {
  const IndexT idxLive; // Inattainable index.
  static constexpr unsigned int noPath = NodePath::pathMax();
  static constexpr unsigned int maskExtinct = noPath;
  static constexpr unsigned int maskLive = maskExtinct - 1;
  static constexpr unsigned int relMax = 1ul << 15;

  vector<IndexT> smIdx; // Root- or node-relative SampleMap index.
  vector<PathT> pathFront;  // Paths reaching the frontier.

  /**
     @brief Setter for path reaching an index.

     @param idx is the index in question.

     @param path is the reaching path.
   */
  void set(IndexT idx,
		  unsigned int path = maskExtinct) {
    pathFront[idx] = path;
  }


  void set(IndexT idx,
                  PathT path,
                  IndexT smIdx) {
    pathFront[idx] = path;
    this->smIdx[idx] = smIdx;
  }


  /**
     @brief Determines a sample's coordinates with respect to the front level.

     @param idx is the node-relative index of the sample.

     @param path outputs the path offset of the sample in the front level.

     @return true iff contents copied.
   */
  bool copyLive(IdxPath *backRef,
                       IndexT idx,
                       IndexT backIdx) const {
    if (!isLive(idx)) {
      return false;
    }
    else {
      backRef->set(backIdx, pathFront[idx], smIdx[idx]);
      return true;
    }
  }
  
 public:

  IdxPath(IndexT idxLive_);

  /**
     @brief When appropriate, localizes indexing at the cost of
     trebling span of memory accesses:  char (PathT) vs. char + uint16.

     @return true iff node-relative indexing expected to be profitable.
   */
  static bool localizes(IndexT bagCount,
			       IndexT idxMax) {
    return idxMax > relMax || bagCount <= 3 * relMax ? false : true;
  }

  
  IndexT getMapIdx(IndexT idx) const {
    return smIdx[idx];
  }


  /**
     @brief Setter for path reaching an index.

     @param idx is the index in question.

     @param path is the reaching path.
   */
  void setSuccessor(IndexT idx,
                           unsigned int pathSucc) {                       
    set(idx, pathSucc);
  }


  /**
     @brief Accumulates a path bit vector for a live reference.

     @return shift-stamped path if live else fixed extinct mask.
   */
  static unsigned int pathSucc(unsigned int pathPrev,
				      bool sense) {
    return maskLive & ((pathPrev << 1) | (sense ? 0 : 1));
  }


  static void pathLR(unsigned int pathPrev,
                            PathT& pathL,
                            PathT& pathR) {
    pathL = maskLive & (pathPrev << 1);
    pathR = maskLive & ((pathPrev << 1) | 1);
  }


  /**
     @brief Revises path and target for live index.

     @param idx is the entry index.

     @param path is the revised path.

     @param smIdx is nascent SampleMap index:  only read node-relative.
  */
  void setLive(IndexT idx,
                      PathT path,
                      IndexT smIdx) {
    set(idx, path, smIdx);
  }


  /**
     @brief Marks path as extinct, sets front index to inattainable value.
     Other values undefined.

     @param idx is the index in question.
   */
  void setExtinct(IndexT idx) {
    set(idx, maskExtinct, idxLive);
  }


  /**
     @brief Indicates whether path reaching index is live.

     @param idx is the index in question.

     @return true iff reaching path is not extinct.
   */
  bool isLive(IndexT idx) const {
    return (pathFront[idx] & maskExtinct) == 0;
  }


  /**
     @brief Obtains front-layer path for an index.

     @param idx indexes the path.

     @param pathMask is obtained from the ancestor layer.

     @param[out] is the succesor path.

     @return true iff path is live.
   */
  bool pathSucc(IndexT idx,
		       unsigned int pathMask,
		       PathT& path) const {
    bool isLive = this->isLive(idx);
    path = isLive ? pathFront[idx] & pathMask : noPath;
    return isLive;
  }

  /**
     @brief Determines whether indexed path is live and looks up
     corresponding front index.

     @param idx is the element index.

     @param front outputs the front index, if live.

     @return true iff path live.
   */
  bool frontLive(IndexT idx,
			IndexT &front) const {
    front = smIdx[idx];
    return isLive(idx);
  }

  
  /**
     @brief Resets front coordinates using first level's map.

     @param one2Front maps first level's coordinates to front.
   */
  void backdate(const IdxPath* one2Front) {
    for (IndexT idx = 0; idx < idxLive; idx++) {
      IndexT oneIdx;
      if (frontLive(idx, oneIdx)) {
        if (!one2Front->copyLive(this, oneIdx, idx)) {
	  setExtinct(idx);
	}
      }
    }
  }

};

#endif
