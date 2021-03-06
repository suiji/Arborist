// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file cutset.cc

   @brief Manages workspace of numerical accumulators.

   @author Mark Seligman
 */

#include "splitnux.h"
#include "cutaccum.h"
#include "cutset.h"
#include "splitfrontier.h"
#include "obspart.h"
#include "residual.h"


CutSet::CutSet() {
}


CutSig CutSet::getCut(const SplitNux& nux) const {
  return cutSig[nux.getAccumIdx()];
}


CutSig CutSet::getCut(IndexT accumIdx) const {
  return cutSig[accumIdx];
}


IndexT CutSet::addCut(const SplitFrontier* splitFrontier,
		      const SplitNux* cand) {
  cutAccum.emplace_back(cand, splitFrontier);
  cutSig.emplace_back(cand->getRange());
  return cutSig.size() - 1;
}


void CutSet::setCut(IndexT accumIdx, const CutSig& sig) {
  cutSig[accumIdx] = sig;
}


bool CutSet::leftCut(const SplitNux* nux) const {
  return cutSig[nux->getAccumIdx()].cutLeft;
}


void CutSet::setCutSense(IndexT cutIdx, bool sense) {
  cutSig[cutIdx].cutLeft = sense;
}


double CutSet::getQuantRank(const SplitNux* nux) const {
  return cutSig[nux->getAccumIdx()].quantRank;
}


IndexT CutSet::getIdxRight(const SplitNux* nux) const {
  return cutSig[nux->getAccumIdx()].idxRight;
}


IndexT CutSet::getIdxLeft(const SplitNux* nux) const {
  return cutSig[nux->getAccumIdx()].idxLeft;
}


IndexT CutSet::getImplicitTrue(const SplitNux* nux) const {
  return cutSig[nux->getAccumIdx()].implicitTrue;
}


void CutSet::write(const SplitNux* nux, const CutAccum* accum) {
  if (accum->info > nux->getInfo()) {
    cutSig[nux->getAccumIdx()].write(nux, accum);
  }
}


void CutSig::write(const SplitNux* nux,
		   const CutAccum* accum) {
  idxLeft = accum->idxLeft;
  idxRight = accum->idxRight;
  implicitTrue = accum->lhImplicit(nux);
  quantRank = accum->interpolateRank(nux);
}
