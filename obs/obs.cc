// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file obs.cc

   @brief Methods to set and unset observation cell packing parameters.

   @author Mark Seligman
 */

#include "obs.h"
#include "splitnux.h"


IndexT Obs::maxSCount = 0;
unsigned int Obs::ctgMask = 0;
unsigned int Obs::multLow = 0;
unsigned int Obs::multMask = 0;

double Obs::scale = 1.0;
double Obs::recipScale = 1.0;

  /**
     @brief Sets internal packing parameters.
   */
void Obs::setShifts(IndexT maxSCount_,
		    unsigned int ctgBits,
		    unsigned int multBits) {
  maxSCount = maxSCount_;
  multLow = ctgLow + ctgBits;
  multMask = (1ul << multBits) - 1;
  ctgMask = (1ul << ctgBits) - 1;
}


void Obs::setScale(double yMax) {
  scale = (yMax * maxSCount) / 0.49;
  if (scale < 1.0)
    scale = 1.0;
  recipScale = 1.0 / scale;
}


void Obs::deImmutables() {
  maxSCount = multMask = ctgMask = 0;
  scale = recipScale = 1.0;
}


ObsReg Obs::residualReg(const Obs* obsCell,
			const SplitNux* nux) {
  ObsReg total = regTotal(obsCell + nux->getObsStart(), nux->getObsExtent());
  return ObsReg(nux->getSum() - total.ySum, nux->getSCount() - total.sCount);
}


void Obs::residualCtg(const Obs* obsCell,
		      const SplitNux* nux,
		      double& sum,
		      IndexT& sCount,
		      vector<double>& ctgImpl) {
  ctgResidual(obsCell + nux->getObsStart(), nux->getObsExtent(), sum, sCount, &ctgImpl[0]);
}
