// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file splitfrontier.cc

   @brief Methods to implement CART-specific splitting of frontier.

   @author Mark Seligman
 */


#include "frontier.h"
#include "sfcart.h"
#include "splitnux.h"
#include "trainframe.h"
#include "ompthread.h"
#include "splitcart.h"
#include "branchsense.h"


SFRegCart::SFRegCart(Frontier* frontier) :
  SFReg(frontier, false, EncodingStyle::trueBranch) {
}


SFCtgCart::SFCtgCart(Frontier* frontier) :
  SFCtg(frontier, false, EncodingStyle::trueBranch) {
}


/**
   @brief Run objects should not be deleted until after splits have been consumed.
 */
void SFRegCart::clear() {
  SplitFrontier::clear();
}


SFRegCart::~SFRegCart() {
}


SFCtgCart::~SFCtgCart() {
}


void SFCtgCart::clear() {
  SplitFrontier::clear();
}


SplitStyle SFCtgCart::getFactorStyle() const {
  return nCtg == 2 ? SplitStyle::slots : SplitStyle::bits;
}

  
  /**
     @return enumeration indicating slot-style encoding.
   */
SplitStyle SFRegCart::getFactorStyle() const {
  return SplitStyle::slots;
}

  
void SFCtgCart::layerPreset() {
  layerInitSumR(frame->getNPredNum());
  ctgSum = vector<vector<double> >(nSplit);

  sumSquares = frontier->sumsAndSquares(ctgSum);
}


void SFCtgCart::layerInitSumR(PredictorT nPredNum) {
  if (nPredNum > 0) {
    ctgSumAccum = vector<double>(nPredNum * nCtg * nSplit);
    fill(ctgSumAccum.begin(), ctgSumAccum.end(), 0.0);
  }
}


void SFRegCart::split(vector<IndexSet>& indexSet,
		      vector<SplitNux>& sc) {
  OMPBound splitTop = sc.size();
#pragma omp parallel default(shared) num_threads(OmpThread::nThread)
  {
#pragma omp for schedule(dynamic, 1)
    for (OMPBound splitPos = 0; splitPos < splitTop; splitPos++) {
      split(&sc[splitPos]);
    }
  }

  nuxMax = maxCandidates(indexSet, sc);
}


void SFCtgCart::split(vector<IndexSet>& indexSet,
		      vector<SplitNux>& sc) {
  OMPBound splitTop = sc.size();
#pragma omp parallel default(shared) num_threads(OmpThread::nThread)
  {
#pragma omp for schedule(dynamic, 1)
    for (OMPBound splitPos = 0; splitPos < splitTop; splitPos++) {
      split(&sc[splitPos]);
    }
  }

  nuxMax = maxCandidates(indexSet, sc);
}


void SFCtgCart::split(SplitNux* cand) {
  SplitCart::splitCtg(this, cand);
}


void SFRegCart::split(SplitNux* cand) {
  SplitCart::splitReg(this, cand);
}
