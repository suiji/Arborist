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


#include "defmap.h"
#include "frontier.h"
#include "sfcart.h"
#include "splitnux.h"
#include "trainframe.h"
#include "ompthread.h"
#include "splitcart.h"
#include "branchsense.h"
#include "runaccum.h"
#include "accumcart.h"


SFRegCart::SFRegCart(Frontier* frontier) :
  SFReg(frontier, false, EncodingStyle::trueBranch, SplitStyle::slots, static_cast<void (SplitFrontier::*) (BranchSense*)>(&SFRegCart::split)) {
}


SFCtgCart::SFCtgCart(Frontier* frontier) :
  SFCtg(frontier, false, EncodingStyle::trueBranch, frontier->getNCtg() == 2 ? SplitStyle::slots : SplitStyle::bits, static_cast<void (SplitFrontier::*) (BranchSense*)>(&SFCtgCart::split)) {
}

  
void SFCtgCart::frontierPreset() {
  for (IndexT splitIdx = 0; splitIdx < nSplit; splitIdx++)
    nodeInfo[splitIdx] = getPreinfo(splitIdx);
}


void SFRegCart::frontierPreset() {
  SFReg::frontierPreset();
  for (IndexT splitIdx = 0; splitIdx < nSplit; splitIdx++)
    nodeInfo[splitIdx] = getPreinfo(splitIdx);
}


double SFRegCart::getPreinfo(IndexT splitIdx) const {
  return (frontier->getSum(splitIdx) * frontier->getSum(splitIdx)) / frontier->getSCount(splitIdx);
}



double SFCtgCart::getPreinfo(IndexT splitIdx) const {
  return sumSquares[splitIdx] / frontier->getSum(splitIdx);
}


void SFRegCart::split(BranchSense* branchSense) {
  vector<SplitNux> sc = defMap->getCandidates(this);
  OMPBound splitTop = sc.size();
#pragma omp parallel default(shared) num_threads(OmpThread::nThread)
  {
#pragma omp for schedule(dynamic, 1)
    for (OMPBound splitPos = 0; splitPos < splitTop; splitPos++) {
      split(&sc[splitPos]);
    }
  }

  maxSimple(sc, branchSense);
}


void SFCtgCart::split(BranchSense* branchSense) {
  vector<SplitNux> sc = defMap->getCandidates(this);
  OMPBound splitTop = sc.size();
#pragma omp parallel default(shared) num_threads(OmpThread::nThread)
  {
#pragma omp for schedule(dynamic, 1)
    for (OMPBound splitPos = 0; splitPos < splitTop; splitPos++) {
      split(&sc[splitPos]);
    }
  }

  maxSimple(sc, branchSense);
}


void SFCtgCart::split(SplitNux* cand) {
  if (cand->isFactor(this)) {
    RunAccum::split(this, cand);
  }
  else {
    CutAccumCtgCart::split(this, cand);
  }
}


void SFRegCart::split(SplitNux* cand) {
  if (cand->isFactor(this)) {
    RunAccum::split(this, cand);
  }
  else {
    CutAccumRegCart::split(this, cand);
  }
}
