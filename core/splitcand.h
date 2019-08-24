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
  const SplitCoord splitCoord; // Node, predictor coordinates.
  const unsigned int sCount;  // Tree node property.
  const double sum; // Tree node property.
  const unsigned char bufIdx; // Per coordinate.  Persists to replay.
  double info; // Tracks during splitting.

  // Initialized or reset after candidate sampling:
  unsigned int setIdx;  // Per coord.
  IndexRange idxRange; // Per coordinate:  post restage.
  unsigned int implicit;  // Per coord:  post restage.

  // Copied to SplitNux, if arg-max:
  //
  unsigned int lhSCount; // # samples subsumed by split LHS:  > 0 iff split.
  unsigned int lhExtent; // Index count of split LHS.
  unsigned int lhImplicit; // LHS implicit index count:  numeric only.

  // Copied to decision node, if arg-max.  Numeric only:
  //
  IndexRange rankRange;

  /**
     @brief decrements 'info' value by information of parent node.

     @return true iff net information gain over parent.
   */
  bool infoGain(const class SplitFrontier*);
  
  /**
     @brief Writes the left-hand characterization of a cut-based split.

     @param spNode contains the tree-node summary.

     @param accum contains the split characterization.
   */
  void writeNum(const class SplitFrontier* spNode,
                const class SplitAccum& accum);
  
public:

  SplitCand(const class SplitFrontier* splitNode,
            const class Frontier* index,
            const SplitCoord& splitCoord_,
            unsigned int bufIdx_,
            unsigned int noSet);

  auto getInfo() const {
    return info;
  }
  
  /**
     @brief Resets trial information value of this greater.

     @param[out] runningMax holds the running maximum value.

     @return true iff value revised.
   */
  bool maxInfo(double& runningMax) const {
    if (info > runningMax) {
      runningMax = info;
      return true;
    }
    return false;
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
    return idxRange.getStart();
  }


  /**
     @brief Accessor for cell upper index.
   */
  auto getIdxEnd() const {
    return idxRange.getEnd() - 1;
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


  auto getLhSCount() const {
    return lhSCount;
  }

  auto getLhExtent() const {
    return lhExtent;
  }

  auto getLhImplicit() const {
    return lhImplicit;
  }

  auto getIdxRange() const {
    return idxRange;
  }

  auto getRankRange() const {
    return rankRange;
  }
  
  /**
     @return Count of indices in cell:  equals node size iff no implicit
     indices.
   */
  auto getExtent() const {
    return getIdxEnd()- getIdxStart() + 1;
  }

  /**
     @brief Retains split coordinate iff target is not a singleton.  Pushes
     back run counts, if applicable.

     @param sg holds partially-initialized split coordinates.

     @param[in, out] runCount accumulates nontrivial run counts.

 `   @param[in, out] sc2 accumulates "actual" splitting coordinates.

     @return true iff candidate remains splitable.
  */
  bool schedule(const class Level *levelFront,
		const class Frontier *indexLevel,
		vector<unsigned int> &runCount);

  /**
     @brief Initializes field values known only following restaging.

     Entry singletons should not reach here.
  */
  void initLate(const class Level* levelFront,
		const class Frontier* iLevel,
                vector<unsigned int>& runCount,
		unsigned int rCount);

  
  void split(const class SFReg *spReg,
	     const class ObsPart *samplePred);


  void split(class SFCtg *spCtg,
	     const class ObsPart *samplePred);

  /**
     @brief Main entry for classification numerical split.
   */
  void splitNum(class SFCtg *spCtg,
                const class SampleRank spn[]);

  /**
     @brief Main entry for regression numerical split.
   */
  void splitNum(const class SFReg *spReg,
                const class SampleRank spn[]);

  void numCtgDense(class SFCtg *spCtg,
                   const class SampleRank spn[]);

  void numCtgGini(SFCtg *spCtg,
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

  void splitFac(const class SFReg *spReg,
                const class SampleRank spn[]);

  void splitFac(class SFCtg *spCtg,
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
  void splitRuns(class SFCtg *spCtg);


  /**
     @brief Adapated from splitRuns().  Specialized for two-category case in
     which LH subsets accumulate.  This permits running LH 0/1 sums to be
     maintained, as opposed to recomputed, as the LH set grows.

     @param spCtg is a categorical response summary.
  */
  void splitBinary(class SFCtg *spCtg);


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
  void buildRuns(class SFCtg *spCtg,
                 const SampleRank spn[]) const;

  /**
     @brief Writes the left-hand characterization of a factor-based
     split with numerical or binary response.

     @param runSet organizes responsed statistics by factor code.

     @param cut is the LHS/RHS separator position in the vector of
     factor codes maintained by the run-set.
   */
  void writeSlots(const class SplitFrontier *splitNode,
                  class RunSet *runSet,
                  unsigned int cut);

  /**
     @brief Writes the left-hand characterization of a factor-based
     split with categorical response.

     @param lhBits is a compressed representation of factor codes for the LHS.
   */
  void writeBits(const class SplitFrontier *sp,
                 unsigned int lhBits);
};

#endif
