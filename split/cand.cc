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


#include "deffrontier.h"
#include "cand.h"


void Cand::precandidates(DefFrontier* defFrontier) {
  // TODO:  Preempt overflow by walking wide subtrees depth-nodeIdx.
  for (IndexT splitIdx = 0; splitIdx < defFrontier->getNSplit(); splitIdx++) {
    if (!defFrontier->isUnsplitable(splitIdx)) { // Node can split.
      for (PredictorT predIdx = 0; predIdx < defFrontier->getNPred(); predIdx++) {
	(void) defFrontier->preschedule(SplitCoord(splitIdx, predIdx));
      }
    }
  }
}
