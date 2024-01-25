// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file sfcart.cc

   @brief CART-specific splitting along frontier.

   @author Mark Seligman
 */


#include "frontier.h"
#include "sfcart.h"
#include "splitnux.h"
#include "splitcart.h"
#include "branchsense.h"
#include "runaccum.h"
#include "cutaccumcart.h"


SFRegCart::SFRegCart(Frontier* frontier) :
  SFReg(frontier, false, EncodingStyle::trueBranch, SplitStyle::slots, &SplitFrontier::splitSimple, static_cast<void (SplitFrontier::*) (SplitNux&)>(&SFRegCart::split)) {
}


SFCtgCart::SFCtgCart(Frontier* frontier) :
  SFCtg(frontier, false, EncodingStyle::trueBranch, frontier->getNCtg() == 2 ? SplitStyle::slots : SplitStyle::bits, &SplitFrontier::splitSimple, static_cast<void (SplitFrontier::*) (SplitNux&)>(&SFCtgCart::split)) {
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
