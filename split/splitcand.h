// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#ifndef SPLIT_SPLITCAND_H
#define SPLIT_SPLITCAND_H

/**
   @file splitcand.h

   @brief Class definition for splitting candidate representation.

   @author Mark Seligman

 */
#include "splitnux.h"
#include "splitcoord.h"
#include "typeparam.h"

#include <memory>
#include <vector>

/**
   @brief Encapsulates information needed to drive splitting.
 */
class SplitCand {
  const IndexT sCount;  // Tree node property.
  const double sum; // Tree node property.
  IndexT implicitCount;  // Per coord:  post restage.

  SplitNux splitNux; // Copied out on argmax.

  
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
            const SplitCoord& splitCoord,
            unsigned int bufIdx_,
            IndexT noSet);


  ~SplitCand() {
  }


  auto getSplitNux() const {
    return splitNux;
  }

  
  auto getInfo() const {
    return splitNux.info;
  }
  
  /**
     @brief Resets trial information value of this greater.

     @param[out] runningMax holds the running maximum value.

     @return true iff value revised.
   */
  bool maxInfo(double& runningMax) const {
    if (splitNux.info > runningMax) {
      runningMax = splitNux.info;
      return true;
    }
    return false;
  }

  auto getSplitCoord() const {
    return splitNux.splitCoord;
  }

  auto getSetIdx() const {
    return splitNux.setIdx;
  }

  auto getBufIdx() const {
    return splitNux.bufIdx;
  }

  /**
     @brief Accessor for cell lower index.
   */
  auto getIdxStart() const {
    return splitNux.idxRange.getStart();
  }


  /**
     @brief Accessor for cell upper index.
   */
  auto getIdxEnd() const {
    return splitNux.idxRange.getEnd() - 1;
  }


  /**
    @brief Accessor for implicit index count.
   */
  auto getImplicitCount() const {
    return implicitCount;
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
    return splitNux.lhSCount;
  }

  auto getLhExtent() const {
    return splitNux.lhExtent;
  }

  auto getLhImplicit() const {
    return splitNux.lhImplicit;
  }

  auto getIdxRange() const {
    return splitNux.idxRange;
  }

  auto getRankRange() const {
    return splitNux.rankRange;
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

  
  void split(const class SFReg* spReg);


  void split(class SFCtg* spCtg);

  /**
     @brief Main entry for classification numerical split.
   */
  void splitNum(class SFCtg* spCtg);


  /**
     @brief Main entry for regression numerical split.
   */
  void splitNum(const class SFReg* spReg);

  void numCtgDense(class SFCtg* spCtg,
                   const class SampleRank spn[]);

  void numCtgGini(SFCtg *spCtg,
                  const class SampleRank spn[],
                  IndexT idxInit,
                  IndexT idxFinal,
                  IndexT& sCountL,
                  IndexT& rkRight,
                  double& sumL,
                  double& ssL,
                  double& ssR,
                  IndexT& rankLH,
                  IndexT& rankRH,
                  IndexT& rhMin);

  void splitFac(const class SFReg *spReg);

  
  void splitFac(class SFCtg *spCtg);

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
  PredictorT heapSplit(class RunSet *runSet);


  /**
     @brief Builds categorical runs.  Very similar to regression case, but
     the runs also resolve response sum by category.
  */
  void buildRuns(class SFCtg *spCtg) const;

  /**
     @brief Writes the left-hand characterization of a factor-based
     split with numerical or binary response.

     @param runSet organizes responsed statistics by factor code.

     @param cutSlot is the LHS/RHS separator position in the vector of
     factor codes maintained by the run-set.
   */
  void writeSlots(const class SplitFrontier *splitNode,
                  class RunSet *runSet,
                  PredictorT cutSlot);

  /**
     @brief Writes the left-hand characterization of a factor-based
     split with categorical response.

     @param lhBits is a compressed representation of factor codes for the LHS.
   */
  void writeBits(const class SplitFrontier *sp,
                 PredictorT lhBits);
};

#endif
