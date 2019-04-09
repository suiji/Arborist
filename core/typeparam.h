// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file typetypeparam.h

   @brief Definitions for parameterization of internal types and classes.

   @author Mark Seligman

 */


#ifndef ARBORIST_TYPEPARAM_H
#define ARBORIST_TYPEPARAM_H

#include <memory>
#include <utility>

using namespace std;

// Type for caching front-end values, but not necessarily for arithmetic.
typedef float FltVal;

// Floating accumulator type, viz. arithmetic.
typedef double FltAccum;

struct RankRange {
  unsigned int rankLow;
  unsigned int rankHigh;

  void set(unsigned int rankLow,
           unsigned int rankHigh) {
    this->rankLow = rankLow;
    this->rankHigh = rankHigh;
  }
};


typedef unsigned char PathT;

/**
   @brief Split/predictor coordinate pair.
 */
typedef pair<unsigned int, unsigned int> SPPair;

#endif
