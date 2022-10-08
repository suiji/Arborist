// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file runsig.cc

   @brief Methods for maintaining runs of factor-valued predictors after splitting.

   @author Mark Seligman
 */

#include "bv.h"
#include "runsig.h"
#include "splitfrontier.h"
#include "splitnux.h"
#include "interlevel.h"


RunSig::RunSig(vector<RunNux> runNux_,
	       PredictorT splitToken_,
	       PredictorT runsSampled_) :
  runNux(std::move(runNux_)),
    splitToken(splitToken_),
    runsSampled(runsSampled_),
    baseTrue(0),
    runsTrue(0),
    implicitTrue(0),
    runSup(0) {
  }

  
vector<IndexRange> RunSig::getTopRange(const CritEncoding& enc) const {
  vector<IndexRange> rangeVec;
  rangeVec.push_back(IndexRange(getBounds(enc.trueEncoding() ? runsTrue - 1 : runNux.size() - 1)));
  return rangeVec;
}


void RunSig::setTrueBits(const InterLevel* interLevel,
			   const SplitNux& nux,
			   BV* splitBits,
			   size_t bitPos) const {
  for (PredictorT trueIdx = baseTrue; trueIdx < baseTrue + runsTrue; trueIdx++) {
    IndexT code = interLevel->getCode(nux, getObs(trueIdx), nux.isImplicit(runNux[trueIdx]));
    splitBits->setBit(bitPos + code);
  }
}


void RunSig::setObservedBits(const InterLevel* interLevel,
			       const SplitNux& nux,
			       BV* observedBits,
			       size_t bitPos) const {
  for (PredictorT runIdx = 0; runIdx != runsSampled; runIdx++) {
    IndexT code = interLevel->getCode(nux, getObs(runIdx), nux.isImplicit(runNux[runIdx]));
    observedBits->setBit(bitPos + code);
  }
}


vector<IndexRange> RunSig::getRange(const CritEncoding& enc) const {
  PredictorT slotStart, slotEnd;
  if (enc.trueEncoding()) {
    slotStart = baseTrue;
    slotEnd = baseTrue + runsTrue;
  }
  else { // Replay indices explicit on false branch.
    slotStart = baseTrue == 0 ? runsTrue : 0;
    slotEnd = baseTrue == 0 ? runNux.size() : (runNux.size() - runsTrue);
  }
  return getRange(slotStart, slotEnd);
}




vector<IndexRange> RunSig::getRange(PredictorT slotStart,
				      PredictorT slotEnd) const {
  vector<IndexRange> rangeVec(slotEnd - slotStart);
  PredictorT slot = 0;
  for (PredictorT outSlot = slotStart; outSlot != slotEnd; outSlot++) {
    rangeVec[slot++] = getBounds(outSlot);
  }

  return rangeVec;
}


void RunSig::updateCriterion(const SplitNux& cand, SplitStyle style) {
  if (style == SplitStyle::slots) {
    leadSlots(cand);
  }
  else if (style == SplitStyle::bits) {
    leadBits(cand);
  }
  else if (style == SplitStyle::topSlot) {
    topSlot(cand);
  }
}


void RunSig::topSlot(const SplitNux& cand) {
  implicitTrue += getImplicitExtent(cand, runsTrue++);
}


IndexT RunSig::getImplicitExtent(const SplitNux& cand,
				 PredictorT slot) const {
  return cand.isImplicit(runNux[slot]) ? getExtent(slot) : 0;
}

  
void RunSig::leadSlots(const SplitNux& nux) {
  // 'splitToken' is the index of the cut, or highest left slot.
  PredictorT runsLeft = splitToken + 1;
  if (nux.invertTest()) {
    baseTrue = runsLeft;
    runsTrue = runNux.size() - runsLeft;
  }
  else {
    runsTrue = runsLeft;
  }
  for (PredictorT runIdx = baseTrue; runIdx != baseTrue + runsTrue; runIdx++) {
    if (nux.isImplicit(runNux[runIdx])) {
      implicitTrue = getExtent(runIdx);
      break;
    }
  }
}


void RunSig::leadBits(const SplitNux& nux) {
  // Only categories visible to this node can be incorporated into the
  // splitting decision.  By convention, the categories resident in 'true'
  // slots will take the true branch during prediction.  All other categories,
  // regardless whether visible, will take the false branch.  This includes not
  // only categories eclipsed by bagging or conditioning, but also proxy
  // categories not encountered during training, as well as NA.

  // No slot, whether implicit or explicit, should be assigned a branch
  // sense fixed a priori.  Doing so biases predictions for reasons outlined
  // above.  For this reason the true branch is randomly assigned to either
  // the argmax slot subset or its complement.  Because factor splitting is
  // expressed via set membership, the randomization can be performed during
  // training.

  PredictorT lhBits = nux.invertTest() ? slotComplement(splitToken) : splitToken;
  implicitTrue = 0;

  // Places true-sense runs to the left for range and code capture.
  // runNux.size() captures all factor levels visible to the cell.
  vector<RunNux> frTemp;
  for (PredictorT runIdx = 0; runIdx != runNux.size(); runIdx++) {
    if (lhBits & (1ul << runIdx)) {
      frTemp.emplace_back(runNux[runIdx]);
      if (nux.isImplicit(runNux[runIdx]))
	implicitTrue += getImplicitExtent(nux, runIdx);
    }
  }
  runsTrue = frTemp.size();
  for (PredictorT runIdx = 0; runIdx != runNux.size(); runIdx++) {
    if (!(lhBits & (1ul << runIdx))) {
      frTemp.emplace_back(runNux[runIdx]);
    }
  }

  runNux = frTemp;
}
