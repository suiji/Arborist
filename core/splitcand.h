// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#ifndef CORE_SPLITCAND_H
#define CORE_SPLITCAND_H

/**
   @file splitcand.h

   @brief Class definition for splitting candidate representation.

   @author Mark Seligman

 */
#include "splitcoord.h"
#include "typeparam.h"
#include <vector>

/**
   @brief Encapsulates information needed to drive splitting.
 */
class SplitCand {
  static constexpr double minRatioDefault = 0.0;
  static double minRatio;

  SplitCoord splitCoord; // Node, predictor coordinates.  Persists to replay.
  double info; // Tracks during splitting.
  unsigned int idxStart; // Per node.
  unsigned int sCount;  // Per node.
  double sum; // Per node.
  unsigned int setIdx;  // Per coord.
  unsigned int implicit;  // Per coord:  post restage.
  unsigned int idxEnd; // Per coord:  post restage.
  unsigned char bufIdx; // Per coordinate.  Persists to replay.

  /**
     @brief decrements 'info' value by information of parent node.

     @return true iff net information gain over parent..
   */
  bool infoGain(const class SplitNode*);
  
public:
  // Data members persisting through replay:
  unsigned int lhSCount; // # samples subsumed by split LHS:  > 0 iff split.
  unsigned int lhExtent; // Index count of split LHS.
  unsigned int lhImplicit; // LHS implicit index count:  numeric only.

  // Persists through training:
  IndexRange rankRange; // Numeric only.


  // Information initilaized to zero, all other fields uninitialized:
  SplitCand() : splitCoord(), info(0.0) {}
  
  SplitCand(const SplitCoord& splitCoord_,
            unsigned int bufIdx_,
            unsigned int noSet);
  
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

  auto getSplitCoord() const {
    return splitCoord;
  }

  auto getSetIdx() const {
    return setIdx;
  }

  auto getBufIdx() const {
    return bufIdx;
  }

  /**
     @brief Accessor for cell lower index.
   */
  auto getIdxStart() const {
    return idxStart;
  }


  /**
     @brief Accessor for cell upper index.
   */
  auto getIdxEnd() const {
    return idxEnd;
  }


  /**
    @brief Accessor for implicit index count.
   */
  auto getImplicit() const {
    return implicit;
  }

  /**
     @brief Response sum accessor.
   */
  auto getSum() const {
    return sum;
  }


  auto getSCount() const {
    return sCount;
  }


  /**
     @return Count of indices corresponding to LHS.  Only applies
     to rank-based splits.
   */
  auto getLHExplicit() const {
    return lhExtent - lhImplicit;
  }

  /**
     @return Count of indices in cell:  equals node size iff no implicit
     indices.
   */
  auto getExtent() const {
    return idxEnd - idxStart + 1;
  }
  
  /**
     @return Count of indices corresponding to RHS.  Rank-based splits
     only.
   */  
  auto getRHExplicit() const {
    return getExtent() - getLHExplicit();
  }

  /**
     @return Starting index of an explicit branch.  Defaults to left if
     both branches explicit.  Rank-based splits only.
   */
  auto getExplicitBranchStart() const {
    return lhImplicit == 0 ? idxStart : idxStart + getLHExplicit();
  }


  /**
     @return Extent of an explicit branch.  Defaults to left if both
     branches explicit.  Rank-based splits only.
   */
  auto getExplicitBranchExtent() const {
    return lhImplicit == 0 ? getLHExplicit() : getRHExplicit();
  }
  
  /**
     @return true iff left side has no implicit indices.  Rank-based
     splits only.
   */
  bool leftIsExplicit() const {
    return lhImplicit == 0;
  }


  /**
     @return rank split object.
   */
  IndexRange getRankRange() const {
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
  bool schedule(const class SplitNode *splitNode,
                const class Level *levelFront,
		const class IndexLevel *indexLevel,
		vector<unsigned int> &runCount);

  /**
     @brief Initializes field values known only following restaging.

     Entry singletons should not reach here.
  */
  void initLate(const class SplitNode* splitNode,
                const class Level* levelFront,
		const class IndexLevel* iLevel,
                vector<unsigned int>& runCount,
		unsigned int rCount);

  /**
     @brief Sets splitting fields from dense-adjusted index node contents.

     @param levelFront summarizes the current front level.

     @param iSet is the index node from which to initiliaze.
   */
  void indexInit(const class Level* levelFront,
                 const class IndexLevel* iLeve);

  
  void split(const class SPReg *spReg,
	     const class SamplePred *samplePred);
  void split(class SPCtg *spCtg,
	     const class SamplePred *samplePred);

  /**
     @brief Main entry for classification numerical split.
   */
  void splitNum(class SPCtg *spCtg,
                const class SampleRank spn[]);

  /**
     @brief Main entry for regression numerical split.
   */
  void splitNum(const class SPReg *spReg,
                const class SampleRank spn[]);

  void numCtgDense(class SPCtg *spCtg,
                   const class SampleRank spn[]);

  void numCtgGini(SPCtg *spCtg,
                  const class SampleRank spn[],
                  unsigned int idxInit,
                  unsigned int idxFinal,
                  unsigned int &sCountL,
                  unsigned int &rkRight,
                  double &sumL,
                  double &ssL,
                  double &ssR,
                  unsigned int &rankLH,
                  unsigned int &rankRH,
                  unsigned int &rhMin);

  void splitFac(const class SPReg *spReg,
                const class SampleRank spn[]);

  void splitFac(class SPCtg *spCtg,
		const class SampleRank spn[]);

  /**
     @brief Splits blocks of categorical runs.

     Nodes are now represented compactly as a collection of runs.
     For each node, subsets of these collections are examined, looking for the
     Gini argmax beginning from the pre-bias.

     Iterates over nontrivial subsets, coded by integers as bit patterns.  By
     convention, the final run is incorporated into RHS of the split, if any.
     Excluding the final run, then, the number of candidate LHS subsets is
     '2^(runCount-1) - 1'.

     @param spCtg summarizes categorical response.
  */
  void splitRuns(class SPCtg *spCtg);


  /**
     @brief Adapated from splitRuns().  Specialized for two-category case in
     which LH subsets accumulate.  This permits running LH 0/1 sums to be
     maintained, as opposed to recomputed, as the LH set grows.

     @param spCtg is a categorical response summary.
  */
  void splitBinary(class SPCtg *spCtg);


  /**
     @brief Splits runs sorted by binary heap.

     @param runSet contains all run parameters.

     @return slot index of split
   */
  unsigned int heapSplit(class RunSet *runSet);


  /**
     @brief Builds categorical runs.  Very similar to regression case, but
     the runs also resolve response sum by category.
  */
  void buildRuns(class SPCtg *spCtg,
                 const SampleRank spn[]) const;

  /**
     @brief Writes the left-hand characterization of an order-based
     regression split.

     @param splitInfo is the information gain induced by the split.

     @param splitLHSCount is the sample count of the LHS.

     @param rankLH is the left predictor rank of the split.

     @param rankRH is the right predictor rank of the split.

     @param lhDense is true iff the LHS contains a dense blob.

     @param rhMin is either the minimal index commencing the RHS.
   */
  void writeNum(double splitInfo,
                unsigned int splitLHSCount,
                unsigned int rankLH,
                unsigned int rankRH,
                bool lhDense,
                unsigned int rhMin);

  
  /**
     @brief Writes the left-hand characterization of a factor-based
     split with numerical or binary response.

     @param runSet organizes responsed statistics by factor code.

     @param cut is the LHS/RHS separator position in the vector of
     factor codes maintained by the run-set.
   */
  void writeSlots(const class SplitNode *splitNode,
                  class RunSet *runSet,
                  unsigned int cut);

  /**
     @brief Writes the left-hand characterization of a factor-based
     split with categorical response.

     @param lhBits is a compressed representation of factor codes for the LHS.
   */
  void writeBits(const class SplitNode *sp,
                 unsigned int lhBits);


  /**
     @brief Reports whether potential split be informative with respect to a threshold.

     @param[in, out] minInfo is the information threshold for splitting.

     @param[out] lhSCount is the number of samples in LHS.

     @param[out] lhExtent is the number of indices in LHS.

     @return true iff information content exceeds the threshold.
   */
  bool isInformative(double &minInfo,
                     unsigned int &lhSCount,
                     unsigned int &lhExtent) const {
    if (info > minInfo) {
      minInfo = minRatio * info; // Splitting threshold for succesors.
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
