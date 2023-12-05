// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file samplernux.cc

   @brief Core-specific packing/unpacking of external Sampler representations.

   @author Mark Seligman
 */


#include "sampler.h"
#include "samplenux.h"

#include <algorithm>

vector<vector<SamplerNux>> SamplerNux::unpack(const double samples[],
					      IndexT nSamp,
					      unsigned int nTree,
					      PredictorT nCtg) {
  IndexT maxSCount = 0;
  vector<vector<SamplerNux>> nuxOut(nTree);
  const double* sample = samples;
  for (unsigned int tIdx = 0; tIdx < nTree; tIdx++) {
    IndexT sCountTree = 0;
    while (sCountTree < nSamp) {
      PackedT packed = *sample++;
      IndexT sCount = SamplerNux::getSCount(packed);
      sCountTree += sCount;
      maxSCount = max(sCount, maxSCount);
      nuxOut[tIdx].emplace_back(packed);
    }
    // assert(sCountTree == nSamp)
  }
  SampleNux::setShifts(nCtg, maxSCount);

  return nuxOut;
}
