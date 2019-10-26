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


// Index type:  rows, samples, ranks, run counts.
// Should be wide enough to accommodate values approaching # observations.
typedef unsigned int IndexT; 

// Predictor type:  columns, # runs, caridinalities.
// Should accommodate values approaching # predictors or properties.
typedef unsigned int PredictorT;

// Low/extent pair definining range of indices.
struct IndexRange {
  IndexT idxLow;
  IndexT idxExtent;


  IndexRange() :
    idxLow(0),
    idxExtent(0) {
  }

  IndexRange(IndexT idxLow_,
	     IndexT idxExtent_) :
    idxLow(idxLow_),
    idxExtent(idxExtent_) {
  }


  void adjust(IndexT margin,
              IndexT implicit) {
    idxLow -= margin;
    idxExtent -= implicit;
  }


  IndexT getStart() const {
    return idxLow;
  }
  

  
  IndexT getExtent() const {
    return idxExtent;
  }


  /**
     @return end position.
   */
  IndexT getEnd() const {
    return idxLow + idxExtent;
  }


  /**
     @brief Interpolates an intermediate position.

     @param scale should lie in [0.0, 1.0].

     @return fractional scaled position.
   */
  double interpolate(double scale) const {
    return idxLow + scale * idxExtent;
  }
};


typedef unsigned char PathT;

#endif
