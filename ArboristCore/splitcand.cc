// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file splitcand.cc

   @brief Methods to implement splitting candidates.

   @author Mark Seligman
 */

#include "splitcand.h"
#include "index.h"
#include "splitpred.h"
#include "level.h"
#include "runset.h"
#include "samplenux.h"
#include "samplepred.h"

double SplitCand::minRatio = minRatioDefault;

void SplitCand::immutables(double minRatio) {
  SplitCand::minRatio = minRatio;
}

void SplitCand::deImmutables() {
  minRatio = minRatioDefault;
}


SplitCand::SplitCand(unsigned int splitIdx_,
                       unsigned int predIdx_,
                       unsigned int bufIdx_) :
  info(0.0),
  splitIdx(splitIdx_),
  predIdx(predIdx_),
  bufIdx(bufIdx_),
  lhSCount(0),
  lhImplicit(0) {
}


/**
   @brief Initializes field values known only following restaging.  Entry
   singletons should not reach here.

   @return void
 */
void SplitCand::initLate(const SplitPred *splitPred,
                         const Level *levelFront,
                         const IndexLevel *index,
                         unsigned int vecPos,
                         unsigned int setIdx) {
  this->vecPos = vecPos;
  this->setIdx = setIdx;
  unsigned int extent;
  index->getSplitFields(splitIdx, idxStart, extent, sCount, sum);
  info = splitPred->getPrebias(splitIdx);
  implicit = levelFront->adjustDense(splitIdx, predIdx, idxStart, extent);
  idxEnd = idxStart + extent - 1; // May overflow if singleton:  invalid.
}

bool SplitCand::schedule(const SplitPred *splitPred,
                         const Level *levelFront,
                         const IndexLevel *index,
                         vector<unsigned int> &runCount,
                         vector<SplitCand> &sc2) {
  unsigned int rCount;
  if (levelFront->scheduleSplit(splitIdx, predIdx, rCount)) {
    initLate(splitPred, levelFront, index, sc2.size(), rCount > 1 ? runCount.size() : splitPred->getNoSet());
    if (rCount > 1) {
      runCount.push_back(rCount);
    }
    sc2.push_back(*this);
    return true;
  }
  return false;
}




/**
   @brief  Regression splitting based on type:  numeric or factor.
 */
void SplitCand::split(const SPReg *spReg,
                       const SamplePred *samplePred) {
  if (spReg->isFactor(predIdx)) {
    splitFac(spReg, samplePred->PredBase(predIdx, bufIdx));
  }
  else {
    splitNum(spReg, samplePred->PredBase(predIdx, bufIdx));
  }
}


/**
   @brief Categorical splitting based on type:  numeric or factor.
 */
void SplitCand::split(SPCtg *spCtg,
		       const SamplePred *samplePred) {
  if (spCtg->isFactor(predIdx)) {
    splitFac(spCtg, samplePred->PredBase(predIdx, bufIdx));
  }
  else {
    splitNum(spCtg, samplePred->PredBase(predIdx, bufIdx));
  }
}


void SplitCand::splitFac(const SPCtg *spCtg,
                          const SampleRank spn[]) {
  runsCtg(spCtg, spn);

  if (spCtg->getCtgWidth() == 2) {
    splitBinary(spCtg);
  }
  else {
    splitRuns(spCtg);
  }
}


// The four major classes of splitting supported here are based on either
// Gini impurity or weighted variance.  New variants may be supplied in
// future.


/**
   @brief Weighted-variance splitting method.
 */
void SplitCand::splitFac(const SPReg *spReg,
                          const SampleRank spn[]) {
  runsReg(spReg, spn);
  heapSplit(spReg);
}


/**
   @brief Invokes regression/numeric splitting method, currently only Gini available.

   @param indexSet[] is the vector of index nodes.

   @param nodeBase is the vector of SamplePred nodes for this level.

   @return void.
*/
void SplitCand::splitNum(const SPReg *spReg,
                           const SampleRank spn[]) {
  int monoMode = spReg->MonoMode(vecPos, predIdx);
  if (monoMode != 0) {
    if (implicit > 0) {
      splitNumDenseMono(monoMode > 0, spn, spReg);
    }
    else {
      splitNumMono(spReg, monoMode > 0, spn);
    }
  }
  else {
    if (implicit > 0) {
      splitNumDense(spn, spReg);
    }
    else {
      splitNumExpl(spReg, spn);
    }
  }
}


/**
   @brief Weighted-variance splitting method.

   @return void.
*/
void SplitCand::splitNumExpl(const SPReg *spReg,
                              const SampleRank spn[]) {
  unsigned int rkRight, sampleCount;
  FltVal ySum;
  spn[idxEnd].regFields(ySum, rkRight, sampleCount);
  double sumR = ySum;
  unsigned int sCountL = sCount - sampleCount; // >= 1: counts up to, including, this index.
  unsigned int lhSup = idxEnd; // lhExtent = idxEnd + 1 - idxStart;

  // Walks samples backward from the end of nodes so that ties are not split.
  // Signing values avoids decrementing below zero.
  for (int i = int(idxEnd) - 1; i >= int(idxStart); i--) {
    unsigned int sCountR = sCount - sCountL;
    double sumL = sum - sumR;
    double idxGini = (sumL * sumL) / sCountL + (sumR * sumR) / sCountR;
    unsigned int rkThis;
    spn[i].regFields(ySum, rkThis, sampleCount);
    if (idxGini > info && rkThis != rkRight) {
      lhSCount = sCountL;
      lhSup = i; // lhExtent = i + 1 - idxStart
      info = idxGini;
    }
    sCountL -= sampleCount;
    sumR += ySum;
    rkRight = rkThis;
  }

  lhExtent = lhSup + 1 - idxStart;
  writeNum(spReg, spn[lhSup].getRank(), spn[lhSup + 1].getRank());
}

void SplitCand::writeNum(const SplitPred *splitPred,
                         unsigned int rankL,
                         unsigned int rankR) {
  info -= splitPred->getPrebias(splitIdx);
  if (info > 0.0) {
    writeNum(rankL, rankR);
  }
}


/**
   @brief Experimental.  Needs refactoring.

   @return void.
*/
void SplitCand::splitNumDense(const SampleRank spn[],
                                const SPReg *spReg) {
  unsigned int rankDense = spReg->denseRank(predIdx);
  double sumDense = sum;
  unsigned int sCountDense = sCount;
  unsigned int denseLeft, denseRight;
  unsigned int denseCut = spReg->Residuals(spn, idxStart, idxEnd, rankDense, denseLeft, denseRight, sumDense, sCountDense);

  unsigned int idxNext, idxFinal;
  unsigned int rkRight, sampleCount;
  FltVal ySum;
  if (denseRight) {
    ySum = sumDense;
    rkRight = rankDense;
    sampleCount = sCountDense;
    idxNext = idxEnd;
    idxFinal = idxStart;
  }
  else {
    spn[idxEnd].regFields(ySum, rkRight, sampleCount);
    idxNext = idxEnd - 1;
    idxFinal = denseLeft ? idxStart : denseCut;
  }
  double sumR = ySum;
  unsigned int sCountL = sCount - sampleCount;
  unsigned int rankLH = 0;
  unsigned int rankRH = 0; // Splitting rank bounds.
  unsigned int rhInf = idxEnd + 1;  // Always non-negative.
  for (int i = int(idxNext); i >= int(idxFinal); i--) {
    unsigned int sCountR = sCount - sCountL;
    double sumL = sum - sumR;
    double idxGini = (sumL * sumL) / sCountL + (sumR * sumR) / sCountR;
    unsigned int rkThis;
    spn[i].regFields(ySum, rkThis, sampleCount);
    if (idxGini > info && rkThis != rkRight) {
      lhSCount = sCountL;
      rankLH = rkThis;
      rankRH = rkRight;
      rhInf = i + 1;
      info = idxGini;
    }
    sCountL -= sampleCount;
    sumR += ySum;
    rkRight = rkThis;
  }

  // Evaluates the dense component, if not of highest rank.
  if (denseCut != idxEnd) {
    unsigned int sCountR = sCount - sCountL;
    double sumL = sum - sumR;
    double idxGini = (sumL * sumL) / sCountL + (sumR * sumR) / sCountR;
    if (idxGini > info) {
      lhSCount = sCountL;
      rhInf = idxFinal;
      rankLH = rankDense;
      rankRH = rkRight;
      info = idxGini;
    }
  
    if (!denseLeft) { // Walks remaining indices, if any, with rank below dense.
      sCountL -= sCountDense;
      sumR += sumDense;
      rkRight = rankDense;
      for (int i = idxFinal - 1; i >= int(idxStart); i--) {
	unsigned int sCountR = sCount - sCountL;
	double sumL = sum - sumR;
	double idxGini = (sumL * sumL) / sCountL + (sumR * sumR) / sCountR;
	unsigned int rkThis;
	spn[i].regFields(ySum, rkThis, sampleCount);
	if (idxGini > info && rkThis != rkRight) {
	  lhSCount = sCountL;
	  rhInf = i + 1;
	  rankLH = rkThis;
	  rankRH = rkRight;
	  info = idxGini;
	}
	sCountL -= sampleCount;
	sumR += ySum;
	rkRight = rkThis;
      }
    }
  }

  lhImplicit = rankLH >= rankDense ? implicit : 0;
  lhExtent = rhInf - idxStart + lhImplicit;
  writeNum(spReg, rankLH, rankRH);
}


/**
   @brief TODO:  Merge with counterparts.

   @return void.
*/
void SplitCand::splitNumDenseMono(bool increasing,
                                    const SampleRank spn[],
                                    const SPReg *spReg) {
  unsigned int rankDense = spReg->denseRank(predIdx);
  double sumDense = sum;
  unsigned int sCountDense = sCount;
  unsigned int denseLeft, denseRight;
  unsigned int denseCut = spReg->Residuals(spn, idxStart, idxEnd, rankDense, denseLeft, denseRight, sumDense, sCountDense);

  unsigned int idxNext, idxFinal;
  unsigned int rkRight, sampleCount;
  FltVal ySum;
  if (denseRight) {
    ySum = sumDense;
    rkRight = rankDense;
    sampleCount = sCountDense;
    idxNext = idxEnd;
    idxFinal = idxStart;
  }
  else {
    spn[idxEnd].regFields(ySum, rkRight, sampleCount);
    idxNext = idxEnd - 1;
    idxFinal = denseLeft ? idxStart : denseCut;
  }
  double sumR = ySum;
  unsigned int sCountL = sCount - sampleCount;

  unsigned int rankLH = 0;
  unsigned int rankRH = 0; // Splitting rank bounds.
  unsigned int rhInf = idxEnd + 1;  // Always non-negative.
  for (int i = int(idxNext); i >= int(idxFinal); i--) {
    unsigned int sCountR = sCount - sCountL;
    double sumL = sum - sumR;
    double idxGini = (sumL * sumL) / sCountL + (sumR * sumR) / sCountR;
    unsigned int rkThis;
    spn[i].regFields(ySum, rkThis, sampleCount);
    if (idxGini > info && rkThis != rkRight) {
      bool up = (sumL * sCountR <= sumR * sCountL);
      if (increasing ? up : !up) {
        lhSCount = sCountL;
        rankLH = rkThis;
        rankRH = rkRight;
        rhInf = i + 1;
        info = idxGini;
      }
    }
    sCountL -= sampleCount;
    sumR += ySum;
    rkRight = rkThis;
  }

  // Evaluates the dense component, if not of highest rank.
  if (denseCut != idxEnd) {
    unsigned int sCountR = sCount - sCountL;
    double sumL = sum - sumR;
    double idxGini = (sumL * sumL) / sCountL + (sumR * sumR) / sCountR;
    if (idxGini > info) {
      lhSCount = sCountL;
      rhInf = idxFinal;
      rankLH = rankDense;
      rankRH = rkRight;
      info = idxGini;
    }
  
    if (!denseLeft) {  // Walks remaining indices, if any, with rank below dense.
      sCountL -= sCountDense;
      sumR += sumDense;
      rkRight = rankDense;
      for (int i = idxFinal - 1; i >= int(idxStart); i--) {
	unsigned int sCountR = sCount - sCountL;
	double sumL = sum - sumR;
	double idxGini = (sumL * sumL) / sCountL + (sumR * sumR) / sCountR;
	unsigned int rkThis;
	spn[i].regFields(ySum, rkThis, sampleCount);
	if (idxGini > info && rkThis != rkRight) {
	  bool up = (sumL * sCountR <= sumR * sCountL);
	  if (increasing ? up : !up) {
	    lhSCount = sCountL;
	    rhInf = i + 1;
	    rankLH = rkThis;
	    rankRH = rkRight;
	    info = idxGini;
	  }
	}
	sCountL -= sampleCount;
	sumR += ySum;
	rkRight = rkThis;
      }
    }
  }

  lhImplicit = rankLH >= rankDense ? implicit : 0;
  lhExtent = rhInf - idxStart + lhImplicit;
  writeNum(spReg, rankLH, rankRH);
}


/**
   @brief Weighted-variance splitting method.

   @return void.
*/
void SplitCand::splitNumMono(const SPReg *spReg,
                              bool increasing,
                              const SampleRank spn[]) {
  unsigned int rkRight, sampleCount;
  FltVal ySum;
  spn[idxEnd].regFields(ySum, rkRight, sampleCount);
  double sumR = ySum;
  unsigned int sCountL = sCount - sampleCount; // >= 1: counts up to, including, this index.

  unsigned int lhSup = idxEnd;

  // Walks samples backward from the end of nodes so that ties are not split.
  // Signing values avoids decrementing below zero.
  for (int i = int(idxEnd) - 1; i >= int(idxStart); i--) {
    int sCountR = sCount - sCountL;
    double sumL = sum - sumR;
    double idxGini = (sumL * sumL) / sCountL + (sumR * sumR) / sCountR;
    unsigned int rkThis;
    spn[i].regFields(ySum, rkThis, sampleCount);
    if (idxGini > info && rkThis != rkRight) {
      bool up = (sumL * sCountR <= sumR * sCountL);
      if (increasing ? up : !up) {
        lhSCount = sCountL;
        lhSup = i;
        info = idxGini;
      }
    }
    sCountL -= sampleCount;
    sumR += ySum;
    rkRight = rkThis;
  }

  lhExtent = lhSup - 1 - idxStart;
  writeNum(spReg, spn[lhSup].getRank(), spn[lhSup+1].getRank());
}




void SplitCand::splitNum(SPCtg *spCtg,
                          const SampleRank spn[]) {
  return implicit > 0 ? numCtgDense(spCtg, spn) : numCtg(spCtg, spn);
}


void SplitCand::numCtg(SPCtg *spCtg,
                        const SampleRank spn[]) {
  unsigned int sCountL = sCount;
  unsigned int rkRight = spn[idxEnd].getRank();
  double sumL = sum;
  double ssL = spCtg->getSumSquares(splitIdx);
  double ssR = 0.0;
  unsigned int rankRH = 0;
  unsigned int rankLH = 0;
  unsigned int rhInf = idxEnd;
  lhSCount = numCtgGini(spCtg, spn, idxEnd, idxStart, sCountL, rkRight, sumL, ssL, ssR, rankLH, rankRH, rhInf);

  lhExtent = rhInf - idxStart;
  writeNum(spCtg, rankLH, rankRH);
}


unsigned int SplitCand::numCtgGini(SPCtg *spCtg,
				    const SampleRank spn[],
				    unsigned int idxNext,
				    unsigned int idxFinal,
				    unsigned int &sCountL,
				    unsigned int &rkRight,
				    double &sumL,
				    double &ssL,
				    double &ssR,
				    unsigned int &rankLH,
				    unsigned int &rankRH,
				    unsigned int &rhInf) {
  unsigned int lhSampCt = 0;
  unsigned int numIdx = spCtg->getNumIdx(predIdx);
  // Signing values avoids decrementing below zero.
  for (int idx = int(idxNext); idx >= int(idxFinal); idx--) {
    FltVal ySum;    
    unsigned int yCtg, rkThis;
    unsigned int sampleCount = spn[idx].ctgFields(ySum, rkThis, yCtg);
    FltVal sumR = sum - sumL;
    if (rkThis != rkRight && spCtg->StableDenoms(sumL, sumR)) {
      FltVal cutGini = ssL / sumL + ssR / sumR;
      if (cutGini > info) {
        lhSampCt = sCountL;
	rankLH = rkThis;
	rankRH = rkRight;
	rhInf = idx + 1;
        info = cutGini;
      }
    }
    rkRight = rkThis;

    sCountL -= sampleCount;
    sumL -= ySum;

    double sumRCtg = spCtg->accumCtgSum(splitIdx, numIdx, yCtg, ySum);
    ssR += ySum * (ySum + 2.0 * sumRCtg);
    double sumLCtg = spCtg->getCtgSum(splitIdx, yCtg) - sumRCtg;
    ssL += ySum * (ySum - 2.0 * sumLCtg);
  }

  return lhSampCt;
}


void SplitCand::numCtgDense(SPCtg *spCtg,
                             const SampleRank spn[]) {
  unsigned int rankDense = spCtg->denseRank(predIdx);
  double sumDense = sum;
  unsigned int sCountDense = sCount;
  bool denseLeft, denseRight;
  vector<double> sumDenseCtg;
  unsigned int denseCut = spCtg->Residuals(spn, splitIdx, idxStart, idxEnd, rankDense, denseLeft, denseRight, sumDense, sCountDense, sumDenseCtg);

  unsigned int idxFinal;
  unsigned int sCountL = sCount;
  unsigned int rkRight;
  double sumL = sum;
  double ssL = spCtg->getSumSquares(splitIdx);
  double ssR = 0.0;
  if (denseRight) { // Implicit values to the far right.
    idxFinal = idxStart;
    rkRight = rankDense;
    spCtg->applyResiduals(splitIdx, predIdx, ssL, ssR, sumDenseCtg);
    sCountL -= sCountDense;
    sumL -= sumDense;
  }
  else {
    idxFinal = denseLeft ? idxStart : denseCut + 1;
    rkRight = spn[idxEnd].getRank();
  }

  unsigned int rankRH = 0;
  unsigned int rankLH = 0;
  unsigned int rhInf = idxEnd;
  lhSCount = numCtgGini(spCtg, spn, idxEnd, idxFinal, sCountL, rkRight, sumL, ssL, ssR, rankLH, rankRH, rhInf);

  // Evaluates the dense component, if not of highest rank.
  if (denseCut != idxEnd) {
    FltVal sumR = sum - sumL;
    if (spCtg->StableDenoms(sumL, sumR)) {
      FltVal cutGini = ssL / sumL + ssR / sumR;
      if (cutGini >  info) {
	lhSCount = sCountL;
	rhInf = idxFinal;
	rankLH = rankDense;
	rankRH = rkRight;
	info = cutGini;
      }
    }

    // Walks remaining indices, if any with ranks below dense.
    if (!denseLeft) {
      spCtg->applyResiduals(splitIdx, predIdx, ssR, ssL, sumDenseCtg);
      sCountL -= sCountDense;
      sumL -= sumDense;
      lhSCount = numCtgGini(spCtg, spn, denseCut, idxStart, sCountL, rkRight, sumL, ssL, ssR, rankLH, rankRH, rhInf);
    }
  }

  lhImplicit = rankLH >= rankDense ? implicit : 0;
  lhExtent = rhInf - idxStart + lhImplicit;
  writeNum(spCtg, rankLH, rankRH);
}


/**
   Regression runs always maintained by heap.
*/
void SplitCand::runsReg(const SPReg *spReg,
			 const SampleRank spn[]) const {
  RunSet *runSet = spReg->rSet(setIdx);
  unsigned int rankDense = spReg->denseRank(predIdx);
  double sumHeap = 0.0;
  unsigned int sCountHeap = 0;
  unsigned int rkThis = spn[idxEnd].getRank();
  unsigned int frEnd = idxEnd;

  // Signing values avoids decrementing below zero.
  //
  for (int i = int(idxEnd); i >= int(idxStart); i--) {
    unsigned int rkRight = rkThis;
    unsigned int sampleCount;
    FltVal ySum;
    spn[i].regFields(ySum, rkThis, sampleCount);

    if (rkThis == rkRight) { // Same run:  counters accumulate.
      sumHeap += ySum;
      sCountHeap += sampleCount;
    }
    else { // New run:  flush accumulated counters and reset.
      runSet->write(rkRight, sCountHeap, sumHeap, frEnd - i, i+1);

      sumHeap = ySum;
      sCountHeap = sampleCount;
      frEnd = i;
    }
  }
  
  // Flushes the remaining run.  Also flushes the implicit run, if dense.
  //
  runSet->write(rkThis, sCountHeap, sumHeap, frEnd - idxStart + 1, idxStart);
  if (implicit > 0) {
    runSet->writeImplicit(rankDense, sCount, sum, implicit);
  }
}


/**
   @brief Splits runs sorted by binary heap.

   @param runSet contains all run parameters.

   @param outputs computed split parameters.

   @return initialized LH split signature.
*/
void SplitCand::heapSplit(const SPReg *spReg) {
  RunSet *runSet = spReg->rSet(setIdx);
  runSet->heapMean();
  runSet->dePop();

  unsigned int sCountL = 0;
  double sumL = 0.0;
  int cut = -1; // Top index of lh ords in 'facOrd' (q.v.).
  for (unsigned int outSlot = 0; outSlot < runSet->getRunCount() - 1; outSlot++) {
    unsigned int sCountRun;
    sumL += runSet->sumHeap(outSlot, sCountRun);
    sCountL += sCountRun;
    unsigned int sCountR = sCount - sCountL;
    double sumR = sum - sumL;
    double cutGini = (sumL * sumL) / sCountL + (sumR * sumR) / sCountR;
    if (cutGini > info) {
      info = cutGini;
      cut = outSlot;
    }
  }

  writeSlots(spReg, cut);
}

void SplitCand::writeSlots(const SplitPred *splitPred,
                           int cut) {
  info -= splitPred->getPrebias(splitIdx);
  if (info > 0.0) {
    RunSet *runSet = splitPred->rSet(setIdx);
    lhExtent = runSet->lHSlots(cut, lhSCount);
  }
}


/**
   @brief Builds categorical runs.  Very similar to regression case, but the runs
   also resolve response sum by category.  Further, heap is optional, passed only
   when run count has been estimated to be wide:

*/
void SplitCand::runsCtg(const SPCtg *spCtg,
			 const SampleRank spn[]) const {
  double sumLoc = 0.0;
  unsigned int sCountLoc = 0;
  unsigned int rkThis = spn[idxEnd].getRank();
  auto runSet = spCtg->rSet(setIdx);
  
  
  // Signing values avoids decrementing below zero.
  unsigned int frEnd = idxEnd;
  for (int i = int(idxEnd); i >= int(idxStart); i--) {
    unsigned int rkRight = rkThis;
    unsigned int yCtg;
    FltVal ySum;
    unsigned int sampleCount = spn[i].ctgFields(ySum, rkThis, yCtg);

    if (rkThis == rkRight) { // Current run's counters accumulate.
      sumLoc += ySum;
      sCountLoc += sampleCount;
    }
    else { // Flushes current run and resets counters for next run.
      runSet->write(rkRight, sCountLoc, sumLoc, frEnd - i, i + 1);

      sumLoc = ySum;
      sCountLoc = sampleCount;
      frEnd = i;
    }
    runSet->accumCtg(yCtg, ySum);
  }

  
  // Flushes remaining run.
  runSet->write(rkThis, sCountLoc, sumLoc, frEnd - idxStart + 1, idxStart);
  if (implicit > 0) {
    runSet->writeImplicit(spCtg->denseRank(predIdx), sCount, sum, implicit, spCtg->getColumnSums(splitIdx));
  }
}


/**
   @brief Splits blocks of categorical runs.

   @param sum is the sum of response values for this index node.

   
   @param lhSampCt outputs LHS sample count.

   @return true iff the node splits.

   Nodes are now represented compactly as a collection of runs.
   For each node, subsets of these collections are examined, looking for the
   Gini argmax beginning from the pre-bias.

   Iterates over nontrivial subsets, coded by integers as bit patterns.  By
   convention, the final run is incorporated into the RHS of the split, if any.
   Excluding the final run, then, the number of candidate LHS subsets is
   '2^(runCount-1) - 1'.
*/
void SplitCand::splitRuns(const SPCtg *spCtg) {
  RunSet *runSet = spCtg->rSet(setIdx);
  const unsigned int slotSup = runSet->deWide() - 1;// Uses post-shrink value.
  unsigned int lhBits = 0;
  unsigned int leftFull = (1 << slotSup) - 1;
  // Nonempty subsets as binary-encoded integers:
  for (unsigned int subset = 1; subset <= leftFull; subset++) {
    double sumL = 0.0;
    double ssL = 0.0;
    double ssR = 0.0;
    for (unsigned int yCtg = 0; yCtg < spCtg->getCtgWidth(); yCtg++) {
      double sumCtg = 0.0; // Sum at category 'yCtg' over subset slots.
      for (unsigned int slot = 0; slot < slotSup; slot++) {
	if ((subset & (1 << slot)) != 0) {
	  sumCtg += runSet->getSumCtg(slot, yCtg);
	}
      }
      const double nodeSumCtg = spCtg->getCtgSum(splitIdx, yCtg);
      sumL += sumCtg;
      ssL += sumCtg * sumCtg;
      ssR += (nodeSumCtg - sumCtg) * (nodeSumCtg - sumCtg);
    }
    double sumR = sum - sumL;
    // Only relevant for case weighting:  otherwise sums are >= 1.
    if (spCtg->StableSums(sumL, sumR)) {
      double subsetGini = ssR / sumR + ssL / sumL;
      if (subsetGini > info) {
        info = subsetGini;
        lhBits = subset;
      }
    }
  }

  writeBits(spCtg, lhBits);
}

void SplitCand::writeBits(const SplitPred* splitPred,
                          unsigned int lhBits) {
  info -= splitPred->getPrebias(splitIdx);
  if (info > 0.0) {
    RunSet *runSet = splitPred->rSet(setIdx);
    lhExtent = runSet->lHBits(lhBits, lhSCount);
  }
}


/**
   @brief Adapated from splitRuns().  Specialized for two-category case in
   which LH subsets accumulate.  This permits running LH 0/1 sums to be
   maintained, as opposed to recomputed, as the LH set grows.

   @return true iff the node splits.
 */
void SplitCand::splitBinary(const SPCtg *spCtg) {
  RunSet *runSet = spCtg->rSet(setIdx);
  runSet->heapBinary();
  runSet->dePop();

  const double tot0 = spCtg->getCtgSum(splitIdx, 0); // Sum over category 0.
  const double tot1 = spCtg->getCtgSum(splitIdx, 1); // Sum over category 1.
  double sumL0 = 0.0; // Running sum at category 0.
  double sumL1 = 0.0; // ibid., category 1.
  int cut = -1;
  for (unsigned int outSlot = 0; outSlot < runSet->getRunCount() - 1; outSlot++) {
    bool splitable = runSet->accumBinary(outSlot, sumL0, sumL1);
    FltVal sumL = sumL0 + sumL1;
    FltVal sumR = sum - sumL;
    // sumR, sumL magnitudes can be ignored if no large case/class weightings.
    if (splitable && spCtg->StableDenoms(sumL, sumR)) {
      FltVal ssL = sumL0 * sumL0 + sumL1 * sumL1;
      FltVal ssR = (tot0 - sumL0) * (tot0 - sumL0) + (tot1 - sumL1) * (tot1 - sumL1);
      FltVal cutGini = ssR / sumR + ssL / sumL;
      if (cutGini > info) {
        info = cutGini;
        cut = outSlot;
      }
    } 
  }

  writeSlots(spCtg, cut);
}
