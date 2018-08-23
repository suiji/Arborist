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
   @brief Encapsulates imputed residual values for cut-based (numerical)
   splitting methods.
 */
struct Residual {
  const double sum;  // Imputed response sum over dense indices.
  const unsigned int sCount; // Imputed sample count over dense indices.

  Residual(const class SplitCand* cand,
           double sumTot,
           unsigned int sCountTot);

  void apply(FltVal& ySum,
             unsigned int& sCount) {
    ySum = this->sum;
    sCount = this->sCount;
  }
};


struct ResidualCtg : public Residual {
  const vector<double> ctgImpl; // Imputed response sums, by category.

  ResidualCtg(const class SplitCand *cand,
              double sumTot,
              unsigned int sCountTot,
              const vector<double>& ctgExpl);

  /**
     @brief Applies state from residual encountered to the left.
   */
  void apply(FltVal& ySum,
             unsigned int& sCount,
             double& ssR,
             double& ssL,
             class NumPersistCtg* np);
};

/**
   @brief Persistent workspace for splittting a numerical predictor.

   Cells having implicit dense blobs are split in separate sections,
   calling for a persistent data structure to hold intermediate state.
   NumPersist is tailored for right-to-left index traversal.
 */
class NumPersist {

protected:
  const unsigned int sCount;
  unsigned int sCountL; // running sum of trial LHS sample counts.
  const double sum;
  double sumL; // running sum of trial LHS response.
  const unsigned int rankDense;
  unsigned int cutDense; // Rightmost position beyond implicit blob, if any.
  
  // Read locally but initialized, and possibly reset, externally.
  unsigned int rkThis; // Current rank.
  unsigned int sCountThis; // Current sample count.
  FltVal ySum; // Current response value.
  
  // Revised at each new local maximum of 'info':
  double info; // Information high watermark.  Precipitates split iff > 0.0.
  unsigned int lhSCount; // Sample count of split LHS:  > 0.
  unsigned int rankRH;
  unsigned int rankLH;

  // Minimal RH index.  May be out of bounds:  [0, idxEnd+1].
  unsigned int rhMin;

public:
  NumPersist(const class SplitCand* cand,
             unsigned int rankDense_);

  /**
     @brief Derives LHS statistics and dispatches to candidate.
   */
  void write(class SplitCand* splitCand);
};


/**
   @brief Auxiliary workspace information specific to regression.
 */
class NumPersistReg : public NumPersist {
  const int monoMode; // Presence/direction of monotone constraint.
  const shared_ptr<Residual> resid;
  shared_ptr<Residual> makeResidual(const class SplitCand* cand,
                                    const class SampleRank spn[]);
  /**
     @brief Updates sums-of-squares accumulators as dictated
     by the presence of a residual.
   */
  void leftResidual();


public:
  NumPersistReg(const class SplitCand* splitCand,
                const class SampleRank spn[],
                const class SPReg* spReg);

  /**
     @brief Dispatches appropriate splitting method.
   */
  void split(const class SampleRank spn[],
             unsigned int idxEnd,
             unsigned int idxStart);
  

  /**
     @brief Splits a range of indices having an implicit blob either between
     the two bounds or immediately adjacent to one of them.

     @param resid summarizes the blob's residual statistics.
   */
  void splitImpl(const class SampleRank spn[],
                 unsigned int idxEnd,
                 unsigned int idxStart);


  /**
     @brief Low-level splitting method for explicity index block.
   */
  void splitExpl(const SampleRank spn[],
                 unsigned int idxInit,
                 unsigned int idxFinal);

  /**
     @brief As above, but specialized for monotonicty constraint.
   */
  void splitMono(const SampleRank spn[],
                 unsigned int idxInit,
                 unsigned int idxFinal);
};


/**
   @brief Splitting accumulator for classification.
 */
class NumPersistCtg : public NumPersist {
  const shared_ptr<ResidualCtg> resid;
  double* ctgSum; // Slice of compressed response data structure.
  double* ctgAccum; // Slice of compressed accumulation data structure.
  unsigned int yCtg; // Response category of current index.
  double ssL; // Left sum-of-squares accumulator.
  double ssR; // Right " ".

  shared_ptr<ResidualCtg>
  makeResidual(const class SplitCand* cand,
               const class SampleRank spn[],
               class SPCtg* spCtg);

public:

  NumPersistCtg(const class SplitCand* cand,
                const class SampleRank spn[],
                class SPCtg* spCtg);

  /**
     @brief Dispatches appropriate splitting method.
   */
  void split(const class SampleRank spn[],
             unsigned int idxInit,
             unsigned int idxFinal);

  /**
     @brief Splitting method for categorical response over an explicit
     block of numerical observation indices.

     @param rightCtg indicates whether a category has been set in an
     initialization or previous invocation.
   */
  void splitExpl(const class SampleRank spn[],
                 unsigned int idxInit,
                 unsigned int idxFinal,
                 bool rightCtg = true);

  /**
     @brief As above, but with implicit dense blob.
   */
  void splitImpl(const class SampleRank spn[],
                 unsigned int idxInit,
                 unsigned int idxFinal);

  /**
     @brief Accessor for node-wide sum for a given category.

     @param ctg is the category in question

     @return sum at category over node.
   */
  double getCtgSum(unsigned int ctg) {
    return ctgSum[ctg];
  }


  /**
     @brief Post-increments accumulated sum.

     @param yCtg is the category at which to increment.

     @param sumCtg is the sum by which to increment.
     
     @return value of accumulated sum prior to incrementing.
   */
  double accumCtgSum(unsigned int yCtg,
                     double sumCtg) {
    return ctgAccum[yCtg] != sumCtg;
  }
};


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


  /**
     @brief Sample count accessor.
   */
  auto getSCount() const {
    return sCount;
  }
  
  /**
     @return position in containing vector, if applicable.
   */
  auto getVecPos() const {
    return vecPos;
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
  bool schedule(const class SplitNode *splitNode,
                const class Level *levelFront,
		const class IndexLevel *indexLevel,
		vector<unsigned int> &runCount,
		vector<SplitCand> &sc2);

  void initLate(const class SplitNode *splitNode,
                const class Level *levelFront,
		const class IndexLevel *index,
		unsigned int _splitPos,
		unsigned int _setIdx);

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
     @brief Adapated from splitRuns().  Specialized for two-category case in
     which LH subsets accumulate.  This permits running LH 0/1 sums to be
     maintained, as opposed to recomputed, as the LH set grows.

     @param spCtg is a categorical response summary.
  */
  void splitBinary(class SPCtg *spCtg);

  void splitRuns(class SPCtg *spCtg);



  /**
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

     @param lhBits is a copmressed representation of factor codes
     corresponding the LHS.
   */
  void writeBits(const class SplitNode *sp,
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
