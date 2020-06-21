// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file splitcart.cc

   @brief Directs splitting via accumaltors.

   @author Mark Seligman
 */


#include "accumcart.h"
#include "sfcart.h"
#include "splitnux.h"
#include "runaccum.h"
#include "trainframe.h"
#include "splitcart.h"
#include "frontier.h"


unique_ptr<SplitFrontier> SplitCart::factory(Frontier* frontier) {
  if (frontier->getNCtg() > 0) {
    return make_unique<SFCtgCart>(frontier);
  }
  else {
    return make_unique<SFRegCart>(frontier);
  }
}


void SplitCart::splitReg(const SFRegCart* sf, SplitNux* cand) {
  if (sf->isFactor(cand)) {
    RunAccum *runAccum = sf->getRunAccum(cand->getAccumIdx());
    runAccum->regRuns();
    runAccum->maxVar();
    cand->infoGain(runAccum);
  }
  else {
    CutAccumRegCart numPersist(cand, sf);
    numPersist.split(sf, cand);
    cand->infoGain(&numPersist);
  }
}


void SplitCart::splitCtg(SFCtgCart* sf, SplitNux* cand) {
  if (sf->isFactor(cand)) {
    RunAccum* runAccum = sf->getRunAccum(cand->getAccumIdx());
    runAccum->ctgRuns(sf->getSumSlice(cand));

    if (sf->getNCtg() == 2) {
      runAccum->binaryGini(sf->getSumSlice(cand));
    }
    else {
      runAccum->ctgGini(sf->getSumSlice(cand));
    }
    cand->infoGain(runAccum);
  }
  else {
    CutAccumCtgCart numPersist(cand, sf);
    numPersist.split(sf, cand);
    cand->infoGain(&numPersist);
  }
}



