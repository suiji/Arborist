// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file crit.cc

   @brief Methods implementing generic criteria.

   @author Mark Seligman
 */


#include "crit.h"
#include "trainframe.h"
#include "splitfrontier.h"


void Crit::setQuantRank(const TrainFrame* sf,
			PredictorT predIdx) {
  setNum(sf->interpolate(predIdx, getNumVal()));
}


void Crit::critCut(const SplitNux* nux,
		   const SplitFrontier* splitFrontier) {
  predIdx = nux->getPredIdx();
  val.setNum(splitFrontier->getQuantRank(nux));
}


