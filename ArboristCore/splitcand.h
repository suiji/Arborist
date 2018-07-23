// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#ifndef ARBORIST_SPLITCAND_H
#define ARBORIST_SPLITCAND_H

/**
   @file splitcand.h

   @brief Class definition for splitting candidate representation.

   @author Mark Seligman

 */

#include "typeparam.h"
#include <vector>

/**
   @brief Encapsulates information needed to drive splitting.
 */
class SplitCand {
  static constexpr double minRatioDefault = 0.0;
  static double minRatio;

  double info; // Tracks during splitting.
  unsigned int vecPos; // Position in containing vector.
  unsigned int splitIdx;
  unsigned int predIdx;
  unsigned int idxStart; // Per node.
  unsigned int sCount;  // Per node.
  double sum; // Per node.
  unsigned int setIdx;  // Per pair.
  unsigned int implicit;  // Per pair:  post restage.
  unsigned int idxEnd; // Per pair:  post restage.
  unsigned char bufIdx; // Per pair.

  /**
     @brief Bulk setter method for splits associated with numeric predictor.
     Passes through to generic Init(), with additional rank and implicit-count
     initialization.

     With introduction of dense ranks, splitting ranks can no longer be
     inferred by position alone.  Hence ranks are passed explicitly.

     @return void.
  */
  void inline writeNum(unsigned int rankLow,
		      unsigned int rankHigh) {
    this->rankRange.rankLow = rankLow;
    this->rankRange.rankHigh = rankHigh;
  }


public:
  unsigned int lhSCount; // # samples subsumed by split LHS:  > 0 iff split.
  unsigned int lhExtent; // Index count of split LHS.
  unsigned int lhImplicit; // LHS implicit index count:  numeric only.
  RankRange rankRange; // Numeric only.

  SplitCand() : info(0.0) {}
  
  SplitCand(unsigned int splitIdx_,
            unsigned int predIdx_,
            unsigned int bufIdx_);

  static void immutables(double minRatio_);
  static void deImmutables();

  /**
     @brief Getter for information field.
   */
  auto getInfo() const {
    return info;
  }

  /**
     @brief Setter for information field.
   */
  void setInfo(double info) {
    this->info = info;
  }

  auto getSplitIdx() const {
    return splitIdx;
  }

  auto getPredIdx() const {
    return predIdx;
  }

  auto getSetIdx() const {
    return setIdx;
  }

  auto getBufIdx() const {
    return bufIdx;
  }

  /**
     @return position in containing vector, if applicable.
   */
  auto getVecPos() const {
    return vecPos;
  }

  /**
     @return Starting index of cell.
   */
  auto getLHStart() const {
    return lhImplicit > 0 ? idxStart - lhImplicit + lhExtent : idxStart;
  }

  /**
     @return Extent of cell.
   */
  auto getLHExtent(unsigned int extent) const {
    return lhImplicit > 0 ? extent - lhExtent : lhExtent;
  }

  /**
     @return true iff left side has no implicit indices.
   */
  bool leftIsExplicit() const {
    return lhImplicit == 0;
  }


  RankRange getRankRange() const {
    return rankRange;
  }
  

  /**
     @brief Retains split coordinate iff target is not a singleton.  Pushes
     back run counts, if applicable.

     @param sg holds partially-initialized split coordinates.

     @param[in, out] runCount accumulates nontrivial run counts.

 `   @param[in, out] sc2 accumulates "actual" splitting coordinates.

     @return true iff candidate remains splitable.
  */
  bool schedule(const class SplitPred *splitPred,
                const class Level *levelFront,
		const class IndexLevel *indexLevel,
		vector<unsigned int> &runCount,
		vector<SplitCand> &sc2);

  void initLate(const class SplitPred *splitPred,
                const class Level *levelFront,
		const class IndexLevel *index,
		unsigned int _splitPos,
		unsigned int _setIdx);

  void split(const class SPReg *spReg,
	     const class SamplePred *samplePred);
  void split(class SPCtg *spCtg,
	     const class SamplePred *samplePred);

  void splitNumExpl(const class SPReg *spReg,
                const class SampleRank spn[]);

  void splitNumDense(const class SampleRank spn[],
                     const class SPReg *spReg);

  void splitNumDenseMono(bool increasing,
			 const class SampleRank spn[],
                         const class SPReg *spReg);

  void splitNumMono(const class SPReg *spReg,
                          bool increasing,
                          const class SampleRank spn[]);

  void splitNum(class SPCtg *spCtg,
                const class SampleRank spn[]);

  void splitNum(const class SPReg *spReg,
                const class SampleRank spn[]);

  void numCtgDense(class SPCtg *spCtg,
                   const class SampleRank spn[]);

  void numCtg(class SPCtg *spCtg,
              const class SampleRank spn[]);

  
  unsigned int numCtgGini(SPCtg *spCtg,
			  const class SampleRank spn[],
			  unsigned int idxNext,
			  unsigned int idxFinal,
			  unsigned int &sCountL,
			  unsigned int &rkRight,
			  double &sumL,
			  double &ssL,
			  double &ssR,
			  unsigned int &rankLH,
			  unsigned int &rankRH,
			  unsigned int &rhInf);

  void splitFac(const class SPReg *spReg,
                const class SampleRank spn[]);

  void splitFac(const class SPCtg *spCtg,
		const class SampleRank spn[]);

  void splitBinary(const class SPCtg *spCtg);

  void splitRuns(const class SPCtg *spCtg);

  void heapSplit(const class SPReg *spReg);

  void runsReg(const class SPReg *spReg,
	       const class SampleRank spn[]) const;

  void runsCtg(const class SPCtg *spCtg,
	       const SampleRank spn[]) const;

  void writeNum(const class SplitPred *splitPred,
                unsigned int rankL,
                unsigned int rankR);


  void writeSlots(const class SplitPred *sp,
                  int cut);

  void writeBits(const class SplitPred *sp,
                 unsigned int lhBits);

  /**
     @brief Reports whether potential split be informative with respect to a threshold.

     @param[out] minInfo is the information threshold for successors' splitting.

     @param[out] lhSCount is the number of samples in LHS.

     @param[out] lhExtent is the number of indices in LHS.

     @return true iff information content exceeds the threshold.
   */
  bool isInformative(double &minInfo,
                   unsigned int &lhSCount,
                   unsigned int &lhExtent) const {
    if (info > minInfo) {
      minInfo = minRatio * info;
      lhSCount = this->lhSCount;
      lhExtent = this->lhExtent;
      return true;
    }
    else {
      return false;
    }
  }
};

#endif
