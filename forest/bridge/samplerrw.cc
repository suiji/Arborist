// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file samplerrw.cc

   @brief Core-specific packing/unpacking of external Sampler representations.

   @author Mark Seligman
 */


#include "samplerrw.h"
#include "sampler.h"

vector<vector<SamplerNux>> SamplerRW::unpack(const double samples[],
					     unsigned int nTree,
					     IndexT nSamp) {
  vector<vector<SamplerNux>> nuxOut(nTree);
  const double* sample = samples;
  for (unsigned int tIdx = 0; tIdx < nTree; tIdx++) {
    IndexT sCountTree = 0;
    while (sCountTree < nSamp) {
      PackedT packed = *sample++;
      sCountTree += SamplerNux::getSCount(packed);
      nuxOut[tIdx].emplace_back(packed);
    }
    // assert(sCountTree == nSamp)
  }
  
  return nuxOut;
}
