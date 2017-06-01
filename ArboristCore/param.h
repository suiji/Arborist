// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file param.h

   @brief Definitions for parameterization of classes.

   @author Mark Seligman

 */


#ifndef ARBORIST_PARAM_H
#define ARBORIST_PARAM_H

// Type for caching front-end values, but not necessarily for arithmetic.
typedef float FltVal;


typedef struct {
  unsigned int rankLow;
  unsigned int rankHigh;
} RankRange;


typedef unsigned char PathT;


#endif
