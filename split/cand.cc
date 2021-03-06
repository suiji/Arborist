// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file cand.cc

   @brief Methods building the list of splitting candidates.

   @author Mark Seligman
 */

#include "cand.h"
#include "splitfrontier.h"
#include "defmap.h"
#include "splitcoord.h"

vector<PreCand> Cand::precandidates(SplitFrontier* splitFrontier,
				    DefMap* defMap) {
  // TODO:  Preempt overflow by walking wide subtrees depth-nodeIdx.
  vector<PreCand> preCand;
  for (IndexT splitIdx = 0; splitIdx < splitFrontier->getNSplit(); splitIdx++) {
    if (!splitFrontier->isUnsplitable(splitIdx)) { // Node can split.
      for (PredictorT predIdx = 0; predIdx < splitFrontier->getNPred(); predIdx++) {
	(void) defMap->preschedule(SplitCoord(splitIdx, predIdx), preCand);
      }
    }
  }

  return preCand;
}
