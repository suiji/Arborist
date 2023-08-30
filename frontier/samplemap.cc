// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file samplemap.cc

   @brief Maintains the frontier sample partition.

   @author Mark Seligman
 */


#include "samplemap.h"
#include "sampledobs.h"
#include "pretree.h"


vector<double> SampleMap::scaleSampleScores(const SampledObs* sampledObs,
					    const PreTree* pretree,
					    double scale) const {
  vector<double> sIdx2Score(sampledObs->getBagCount());
  IndexT rangeIdx = 0;
  for (const IndexRange& rg : range) {
    double score = scale * pretree->getScore(ptIdx[rangeIdx]);
    for (IndexT idx = rg.getStart(); idx != rg.getEnd(); idx++) {
      IndexT sIdx = sampleIndex[idx];
      sIdx2Score[sIdx] = score;
    }
    rangeIdx++;
  }

  return sIdx2Score;
}


