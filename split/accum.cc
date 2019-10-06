// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file accum.cc

   @brief Base class for splitting workspace.

   @author Mark Seligman
 */

#include "splitnux.h"
#include "accum.h"
#include "splitnux.h"
#include "obspart.h"
#include "residual.h"


Accum::Accum(const SplitNux* cand,
             IndexT rankDense_) :
  sCount(cand->getSCount()),
  sum(cand->getSum()),
  rankDense(rankDense_),
  sCountL(sCount),
  sumL(sum),
  cutDense(cand->getIdxEnd() + 1),
  info(cand->getInfo()) {
}


IndexT Accum::lhImplicit(const SplitNux* cand) const {
  return rankDense <= rankLH ? cand->getImplicitCount() : 0;
}


unique_ptr<Residual>
Accum::makeResidual(const SplitNux* cand,
                      const SampleRank spn[]) {
  if (cand->getImplicitCount() == 0) {
    return make_unique<Residual>();
  }

  double sumExpl = 0.0;
  IndexT sCountExpl = 0;
  for (int idx = static_cast<int>(cand->getIdxEnd()); idx >= static_cast<int>(cand->getIdxStart()); idx--) {
    IndexT rkThis = spn[idx].regFields(ySum, sCountThis);
    if (rkThis > rankDense) {
      cutDense = idx;
    }
    sCountExpl += sCountThis;
    sumExpl += ySum;
  }
  
  return make_unique<Residual>(sum - sumExpl, sCount - sCountExpl);
}
