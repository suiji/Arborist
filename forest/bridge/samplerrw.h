// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file samplerrw.h

   @brief Core-specific packing/unpacking of external Sampler representations.

   @author Mark Seligman
 */

#ifndef FOREST_BRIDGE_SAMPLERRW_H
#define FOREST_BRIDGE_SAMPLERRW_H

#include "typeparam.h"

#include <vector>
using namespace std;

struct SamplerRW {
  static vector<vector<class SamplerNux>> unpack(const double samples[],
						 IndexT nSamp,
						 unsigned int nTree,
						 PredictorT nCtg = 0);
};

#endif
