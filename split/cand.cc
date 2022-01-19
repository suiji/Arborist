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


#include "defmap.h"
#include "cand.h"
#include "callback.h"


void Cand::precandidates(DefMap* defMap) {
  // TODO:  Preempt overflow by walking wide subtrees depth-nodeIdx.
  IndexT idx = 0;
  vector<double> dRand = CallBack::rUnif(defMap->getNPred() * defMap->getNSplit());
  for (IndexT splitIdx = 0; splitIdx < defMap->getNSplit(); splitIdx++) {
    if (!defMap->isUnsplitable(splitIdx)) { // Node can split.
      for (PredictorT predIdx = 0; predIdx < defMap->getNPred(); predIdx++) {
	(void) defMap->preschedule(SplitCoord(splitIdx, predIdx), dRand[idx++]);
      }
    }
  }
}
