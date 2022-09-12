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
#include "ompthread.h"
#include "splitcart.h"
#include "branchsense.h"
#include "runaccum.h"
#include "cutaccumcart.h"


SFRegCart::SFRegCart(Frontier* frontier) :
  SFReg(frontier, false, EncodingStyle::trueBranch, SplitStyle::slots, static_cast<void (SplitFrontier::*) (vector<SplitNux>&, BranchSense&)>(&SFRegCart::split)) {
}


SFCtgCart::SFCtgCart(Frontier* frontier) :
  SFCtg(frontier, false, EncodingStyle::trueBranch, frontier->getNCtg() == 2 ? SplitStyle::slots : SplitStyle::bits, static_cast<void (SplitFrontier::*) (vector<SplitNux>&, BranchSense&)>(&SFCtgCart::split)) {
}


void SFRegCart::accumPreset() {
  SFReg::accumPreset();
}


void SFRegCart::split(vector<SplitNux>& sc,
		      BranchSense& branchSense) {
  OMPBound splitTop = sc.size();
#pragma omp parallel default(shared) num_threads(OmpThread::nThread)
  {
#pragma omp for schedule(dynamic, 1)
    for (OMPBound splitPos = 0; splitPos < splitTop; splitPos++) {
      split(sc[splitPos]);
    }
  }

  maxSimple(sc, branchSense);
}


void SFCtgCart::split(vector<SplitNux>& sc,
		      BranchSense& branchSense) {
  OMPBound splitTop = sc.size();
#pragma omp parallel default(shared) num_threads(OmpThread::nThread)
  {
#pragma omp for schedule(dynamic, 1)
    for (OMPBound splitPos = 0; splitPos < splitTop; splitPos++) {
      split(sc[splitPos]);
    }
  }

  maxSimple(sc, branchSense);
}


void SFCtgCart::split(SplitNux& cand) {
  if (isFactor(cand)) {
    RunAccumCtg::split(this, runSet.get(), cand);
  }
  else {
    CutAccumCtgCart::split(this, cand);
  }
}


void SFRegCart::split(SplitNux& cand) {
  if (isFactor(cand)) {
    RunAccumReg::split(this, runSet.get(), cand);
  }
  else {
    CutAccumRegCart::split(this, cand);
  }
}
