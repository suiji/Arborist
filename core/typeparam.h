// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file typeparam.h

   @brief Definitions for parameterization of internal types and classes.

   @author Mark Seligman

 */


#ifndef CORE_TYPEPARAM_H
#define CORE_TYPEPARAM_H

#include <memory>
#include <utility>

using namespace std;

// Type for caching front-end values, but not necessarily for arithmetic.
typedef float FltVal;

// Floating accumulator type, viz. arithmetic.
typedef double FltAccum;


// Index type:  rows, samples, ranks.
typedef unsigned int IndexType;

// Low/extent pair definining range of indices.
struct IndexRange {
  IndexType idxLow;
  IndexType idxExtent;

  void set(IndexType idxLow,
           IndexType extent) {
    this->idxLow = idxLow;
    this->idxExtent = extent;
  }


  void adjust(IndexType margin,
              IndexType implicit) {
    idxLow -= margin;
    idxExtent -= implicit;
  }


  IndexType getStart() const {
    return idxLow;
  }
  
  /**
     @return end position.
   */
  IndexType getEnd() const {
    return idxLow + idxExtent;
  }
};


typedef unsigned char PathT;

#endif
