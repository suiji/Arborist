// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#ifndef CART_SPLITACCUM_H
#define CART_SPLITACCUM_H

/**
   @file splitaccum.h

   @brief Accumulator classes for cut-based (numeric) splitting workspaces.

   @author Mark Seligman

 */

#include "typeparam.h"

#include <vector>
using namespace std;


/**
   @brief Encapsulates imputed residual values.
 */
struct Residual {
  const double sum;  // Imputed response sum over dense indices.
  const unsigned int sCount; // Imputed sample count over dense indices.

  /**
     @brief Constructor initializes contents to residual values.

     @param sumExpl is the sum of explicit responses over the cell.

     @param sCountExpl is the sum of explicit sample counts over the cell.
   */
  Residual(double sum_,
           unsigned int sCount_);

  /**
     @brief Outputs residual contents.

     @param[out] ySum outputs the residual response sum.

     @param[out] sCount outputs the residual sample count.
   */  
  void apply(FltVal& ySum,
             unsigned int& sCount) {
    ySum = this->sum;
    sCount = this->sCount;
  }
};


struct ResidualCtg : public Residual {
  const vector<double> ctgImpl; // Imputed response sums, by category.

  ResidualCtg(double sum_,
              unsigned int sCount_,
              const vector<double>& ctgExpl);

  /**
     @brief Applies state from residual encountered to the left.
   */
  void apply(FltVal& ySum,
             unsigned int& sCount,
             double& ssR,
             double& ssL,
             class SplitAccumCtg* np);
};


/**
   @brief Persistent workspace for computing optimal split.

   Cells having implicit dense blobs are split in separate sections,
   calling for a re-entrant data structure to cache intermediate state.
   SplitAccum is tailored for right-to-left index traversal.
 */
class SplitAccum {
protected:
  const unsigned int sCount; // Running sample count along node.
  const double sum; // Running response along node.
  const unsigned int rankDense; // Rank of dense value, if any.
  unsigned int sCountL; // Running sum of trial LHS sample counts.
  double sumL; // Running sum of trial LHS response.
  unsigned int cutDense; // Rightmost position beyond implicit blob, if any.
  
  // Read locally but initialized, and possibly reset, externally.
  unsigned int sCountThis; // Current sample count.
  FltVal ySum; // Current response value.

public:
  // Revised at each new local maximum of 'info':
  double info; // Information high watermark.  Precipitates split iff > 0.0.
  unsigned int lhSCount; // Sample count of split LHS:  > 0.
  unsigned int rankRH; // Maximum rank characterizing split.
  unsigned int rankLH; // Minimum rank charactersizing split.
  unsigned int rhMin; // Min RH index, possibly out of bounds:  [0, idxEnd+1].

  SplitAccum(const class SplitCand* cand,
             unsigned int rankDense_);

  bool lhDense() const {
    return rankDense <= rankLH;
  }
};


/**
   @brief Auxiliary workspace information specific to regression.
 */
class SplitAccumReg : public SplitAccum {
  const int monoMode; // Presence/direction of monotone constraint.
  const shared_ptr<Residual> resid; // Current residual, if any, else null.

  /**
     @brief Creates a residual summarizing implicit splitting state.

     @param cand is the splitting candidate.

     @param spn is the splitting data set.
     
     @return new residual based on the current splitting data set.
   */
  shared_ptr<Residual> makeResidual(const class SplitCand* cand,
                                    const class SampleRank spn[]);
  /**
     @brief Updates accumulators and possibly splits.

     Current rank position assumed to be adjacent to dense rank, whence
     the application of the residual immediately to the left.

     @param rkThis is the rank of the current position.
   */
  void leftResidual(unsigned int rkThis);


public:
  SplitAccumReg(const class SplitCand* splitCand,
                const class SampleRank spn[],
                const class SFReg* spReg);

  /**
     @brief Evaluates trial splitting information as weighted variance.

     @param sumLeft is the sum of responses to the left of a trial split.

     @param sumRight is the sum of responses to the right.

     @param sCountLeft is number of samples to the left.

     @param sCountRight is the number of samples to the right.

     @param info[in, out] outputs max of input and new information 
   */
  static constexpr double infoSplit(double sumLeft,
                                    double sumRight,
                                    unsigned int sCountLeft,
                                    unsigned int sCountRight) {
    return (sumLeft * sumLeft) / sCountLeft + (sumRight * sumRight) / sCountRight;
  }


  static bool infoSplit(double sumLeft,
                        double sumRight,
                        unsigned int sCountLeft,
                        unsigned int sCountRight,
                        double& info) {
    double infoTemp = infoSplit(sumLeft, sumRight, sCountLeft, sCountRight);
    if (infoTemp > info) {
      info = infoTemp;
      return true;
    }
    else {
      return false;
    }
  }


  /**
     @brief Dispatches appropriate splitting method.
   */
  void split(const class SFReg* spReg,
             const class SampleRank spn[],
             class SplitCand* cand);
  

  /**
     @brief Splits a range of indices having an implicit blob either between
     the two bounds or immediately adjacent to one of them.

     @param resid summarizes the blob's residual statistics.
   */
  void splitImpl(const class SampleRank spn[],
                 const class SplitCand* cand);


  /**
     @brief Low-level splitting method for explicit block of indices.
   */
  void splitExpl(const SampleRank spn[],
                 unsigned int rkThis,
                 unsigned int idxInit,
                 unsigned int idxFinal);

  /**
     @brief As above, but specialized for monotonicty constraint.
   */
  void splitMono(const SampleRank spn[],
                 unsigned int rkThis,
                 unsigned int idxInit,
                 unsigned int idxFinal);
};


/**
   @brief Splitting accumulator for classification.
 */
class SplitAccumCtg : public SplitAccum {
  const unsigned int nCtg; // Cadinality of response.
  const shared_ptr<ResidualCtg> resid;
  const vector<double>& ctgSum; // Per-category response sum at node.
  double* ctgAccum; // Slice of compressed accumulation data structure.
  double ssL; // Left sum-of-squares accumulator.
  double ssR; // Right " ".

  /**
     @brief Imputes per-category dense rank statistics as residuals over cell.

     @param cand is the splitting candidate.

     @param spn is the splitting environment.

     @param spCtg summarizes the categorical response.

     @return new residual for categorical response over cell.
  */
  shared_ptr<ResidualCtg>
  makeResidual(const class SplitCand* cand,
               const class SampleRank spn[],
               class SFCtg* spCtg);

public:

  SplitAccumCtg(const class SplitCand* cand,
                const class SampleRank spn[],
                class SFCtg* spCtg);

  /**
     @brief Evaluates trial splitting information as Gini.

     @param ssLeft is the sum of squared responses to the left of a trial split.

     @param ssRight is the sum of squared responses to the right.

     @param sumLeft is the sum of responses to the left.

     @param sumRight is the sum of responses to the right.
   */
  static constexpr double infoSplit(double ssLeft,
                                    double ssRight,
                                    double sumLeft,
                                    double sumRight) {
    return ssLeft / sumLeft + ssRight / sumRight;
  }


  static bool infoSplit(double ssLeft,
                        double ssRight,
                        double sumLeft,
                        double sumRight,
                        double& info) {
    double infoTemp = infoSplit(ssLeft, ssRight, sumLeft, sumRight);
    if (infoTemp > info) {
      info = infoTemp;
      return true;
    }
    else {
      return false;
    }
  }


  /**
     @brief Dispatches appropriate splitting method.
   */
  void split(const class SFCtg* spCtg,
             const class SampleRank spn[],
             class SplitCand* cand);

  
  /**
     @brief Splitting method for categorical response over an explicit
     block of numerical observation indices.

     @param rightCtg indicates whether a category has been set in an
     initialization or previous invocation.
   */
  void splitExpl(const class SampleRank spn[],
                 unsigned int rkThs,
                 unsigned int idxInit,
                 unsigned int idxFinal);

  /**
     @brief As above, but with implicit dense blob.
   */
  void splitImpl(const class SampleRank spn[],
                 const class SplitCand* cand);

  /**
     @brief Accumulates right and left sums-of-squares from
     exposed state.
   */
inline unsigned int stateNext(const class SampleRank spn[],
                         unsigned int idx);

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
    double val = ctgAccum[yCtg];
    ctgAccum[yCtg] += sumCtg;
    return val;
  }
};

#endif
