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

#ifndef ARBORIST_PATH_H
#define ARBORIST_PATH_H

#include <cstdint>
#include <vector>

#include "param.h"

/**
   @brief Records index, start and extent for path reached from MRRA.
 */
class NodePath {
  unsigned int splitIdx; // < noIndex iff path extinct.
  unsigned int idxStart; // Target offset for path.
  unsigned int extent;
  unsigned int relBase; // Dense starting position.
 public:

  static constexpr unsigned int pathMax = 8 * sizeof(PathT) - 1;
  static constexpr unsigned int noPath = 1 << pathMax;

  
  /**
     @brief Sets to non-extinct path coordinates.
   */
  inline void Init(unsigned int _splitIdx, unsigned int _idxStart, unsigned int _extent, unsigned int _relBase) {
    splitIdx = _splitIdx;
    idxStart = _idxStart;
    extent = _extent;
    relBase = _relBase;
  }
  

  inline void Coords(unsigned int &_splitIdx, unsigned int &_idxStart, unsigned int &_extent) const {
    _splitIdx = splitIdx;
    _idxStart = idxStart;
    _extent = extent;
  }

  
  inline unsigned int IdxStart() const {
    return idxStart;
  }


  inline unsigned int Extent() const {
    return extent;
  }
  

  inline unsigned int RelBase() const {
    return relBase;
  }


  inline unsigned int Idx() const {
    return splitIdx;
  }
};


class IdxPath {
  const unsigned int idxLive; // Inattainable index.
  static constexpr unsigned int maskExtinct = NodePath::noPath;
  static constexpr unsigned int maskLive = maskExtinct - 1;
  static constexpr unsigned int relMax = 1 << 15;
  std::vector<unsigned int> relFront;
  std::vector<unsigned char> pathFront;

  // Only defined for enclosing Levels employing node-relative indexing.
  //
  // Narrow for data locality, but wide enough to be useful.  Can
  // be generalized to multiple sizes to accommodate more sophisticated
  // hierarchies.
  //
  std::vector<uint_least16_t> offFront;
 public:

  IdxPath(unsigned int _idxLive);

  /**
     @brief When appropriate, introduces node-relative indexing at the
     cost of trebling span of memory accesses:  char vs. char + uint16.

     @return True iff node-relative indexing expected to be profitable.
   */
  static inline bool Localizes(unsigned int bagCount, unsigned int idxMax) {
    return idxMax > relMax || bagCount <= 3 * relMax ? false : true;
  }

  
  /**
     @bool Accessor for live index count.
   */
  inline unsigned int IdxLive() const {
    return idxLive;
  }


  inline void Set(unsigned int idx, unsigned int path = maskExtinct) {
    pathFront[idx] = path;
  }

  
  /**
   */
  inline void Set(unsigned int idx, unsigned int path, unsigned int relThis, unsigned int ndOff = 0) {
    pathFront[idx] = path;
    relFront[idx] = relThis;
    offFront[idx] = ndOff;
  }


  inline unsigned int RelFront(unsigned int idx) const {
    return relFront[idx];
  }

  
  /**
     @brief Accumulates a path bit vector for a live reference.

     @return shift-stamped path if live else fixed extinct mask.
   */
  inline static unsigned int PathNext(unsigned int pathPrev, bool isLeft) {
    return maskLive & ((pathPrev << 1) | (isLeft ? 0 : 1));
  }
  

  /**
   @brief Revises path for live index.

   @param idx is the current index value.

   @param path is the revised path.

   @return void.
*/
  inline void SetLive(unsigned int idx, unsigned int path) {
    Set(idx, path);
  }


  /**
   @brief Revises path and target for live index.

   @param idx is the current index value.

   @param path is the revised path.

   @param targIdx is the revised relative index.

   @return void.
*/
  inline void SetLive(unsigned int idx, unsigned int path, unsigned int targIdx) {
    Set(idx, path, targIdx);
  }

  
  /**

   @brief Revises path and target for potentially node-relative live index.

   @param idx is the current index value.

   @param path

   @param targIdx is the revised index.

   @return void.
*/
  inline void SetLive(unsigned int idx, unsigned int path, unsigned int targIdx, unsigned int ndOff) {
    Set(idx, path, targIdx, ndOff);
  }

  
  /**
     @brief Marks path as extinct, sets front index to inattainable value.
     Other values undefined.

     @return void.
   */
  inline void SetExtinct(unsigned int idx) {
    Set(idx, maskExtinct, idxLive);
  }


  inline bool IsLive(unsigned idx) const {
    return (pathFront[idx] & maskExtinct) == 0;
  }


  /**
     @brief Caller ensures path is two-valued, with neither lane extinct.

     @return true iff path is LH.
   */
  inline bool IsLeft(unsigned int idx) const {
    return (pathFront[idx] & 0x1) == 0;
  }
  

  inline PathT PathSucc(unsigned int idx, unsigned int pathMask, bool &isLive) const {
    isLive = IsLive(idx);
    return isLive ? pathFront[idx] & pathMask : NodePath::noPath;
  }
  
  
  /**
     @brief Looks up the path leading to the front level and updates
     the index, if either in a switching to a node-relative regime.

     @param idx inputs the path vector index and outputs the index to
     be used in the next level.

     @return path to input index.
   */
  inline unsigned int IdxUpdate(unsigned int &idx, unsigned int pathMask, const unsigned int reachBase[], bool idxUpdate) const {
    bool isLive;
    PathT path = PathSucc(idx, pathMask, isLive);
    if (isLive) {
      // Avoids irregular update unless necessary:
      idx = reachBase != nullptr ? (reachBase[path] + offFront[idx]) : (idxUpdate ? RelFront(idx) : idx);
    }

    return path;
  }


  /**
     @brief Determines whether indexed path is live and looks up
     corresponding front index.

     @param idx is the element index.

     @param front outputs the front index, if live.

     @return true iff path live.
   */
  inline bool FrontLive(unsigned int idx, unsigned int &front) const {
    if (!IsLive(idx)) {
      return false;
    }
    
    front = relFront[idx];
    return true;
  }

  
  /**
     @brief Determines a sample's coordinates with respect to the front level.

     @param idx is the node-relative index of the sample.

     @param path outputs the path offset of the sample in the front level.

     @return true iff contents copied.
   */
  inline bool CopyLive(IdxPath *backRef, unsigned int idx, unsigned int backIdx) const {
    if (!IsLive(idx)) {
      return false;
    }

    backRef->Set(backIdx, pathFront[idx], relFront[idx], offFront[idx]);
    return true;
  }

  
  /**
     @brief Resets front coordinates using first level's map.

     @param one2Front maps first level's coordinates to front.

     @return void.
   */
  inline void Backdate(const IdxPath *one2Front) {
    for (unsigned int idx = 0; idx < idxLive; idx++) {
      unsigned int oneIdx;
      if (FrontLive(idx, oneIdx)) {
        if (!one2Front->CopyLive(this, oneIdx, idx)) {
	  SetExtinct(idx);
	}
      }
    }
  }


};

#endif
