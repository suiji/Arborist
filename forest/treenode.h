// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file cartnode.h

   @brief Data structures and methods implementing CART tree nodes.

   @author Mark Seligman
 */


#ifndef FOREST_TREENODE_H
#define FOREST_TREENODE_H

#include "crit.h"
#include "bv.h"

#include <vector>
#include <complex>



/**
   @brief To replace parallel array access.

   Caches splitting predictor and delta to an explicit branch target.
   Target of complementary branch is computable.  Sense of explicit
   branch and method of computing complement varies with algorithm.
 */
class TreeNode {
  // Initialized from predictor frame or signature:
  static unsigned int rightBits;
  static PredictorT rightMask;

  // Intialized by prediction command:
  static bool trapUnobserved;

  PackedT packed;

protected:

  Crit criterion;
  bool invert;

public:
  /**
     @brief Initializes packing parameters.
   */
  static void initMasks(PredictorT nPred);


  static void initTrap(bool doTrap);

  
  /**
     @brief Resets packing values to default.
   */
  static void deInit();


  static bool trapAndBail();
  

  /**
     @brief Nodes must be explicitly set to non-terminal (delIdx != 0).
   */
  TreeNode() : packed(0ull),
	       criterion(0.0),
	       invert(false) {
  }


  /**
     @brief Constructor for reading complex-encoded nodes.
   */
  TreeNode(complex<double> pair) :
    packed(abs(pair.real())),
    criterion(pair.imag()),
    invert(pair.real() < 0.0) {
  }

  
  /**
     @brief Encodes node contents as complex value.

     @param[out] valOut outputs the complex encoding.
   */
  inline void dump(complex<double>& valOut) const {
    valOut = complex<double>(invert ? -double(packed) : packed, criterion.getVal());
  }

  
  /**
     @brief Sets the invert field to the specified (randomized) sense.
   */
  inline void setInvert(bool invert) {
    this->invert = invert;
  }


  inline void setPredIdx(PredictorT predIdx) {
    packed |= predIdx;
  }
  

  /**
     @brief Getter for splitting predictor.

     @return splitting predictor index.
   */
  inline PredictorT getPredIdx() const {
    return packed & rightMask;
  }


  /**
     @brief Initializes delIdx value; not for resetting.
   */
  inline void setDelIdx(IndexT delIdx) {
    packed |= (size_t(delIdx) << rightBits);
  }
  

  inline void resetDelIdx(IndexT delIdx) {
    packed = getPredIdx();
    packed |= (size_t(delIdx) << rightBits);
  }
  

  /**
     @brief Getter for lh-delta value.

     @return delIdx value.
   */
  inline IndexT getDelIdx() const {
    return packed >> rightBits;
  }


  /**
     @brief Indicates whether node is nonterminal.

     @return True iff delIdx value is nonzero.
   */
  inline bool isNonterminal() const {
    return getDelIdx() != 0;
  }  


  inline bool isTerminal() const {
    return getDelIdx() == 0;
  }


  /**
     @return delta to branch target, given test and inversion sense.
   */
  IndexT delInvert(bool pass) const {
    return getDelIdx() + (invert ? (pass ? 1 : 0) : (pass ? 0 : 1));
  }


  /**
     @brief As above, but ignoring inversion sense.
   */
  IndexT delTest(bool pass) const {
    return getDelIdx() + (pass ? 0 : 1);
  }

  
  void critCut(const class SplitNux& nux,
	       const class SplitFrontier* splitFrontier);


  void critBits(const class SplitNux& nux,
		size_t bitPos);


  /**
     @brief Getter for numeric splitting value.

     @return splitting value.
   */
  inline auto getSplitNum() const {
    return criterion.getNumVal();
  }


  /**
     @return first bit position of split.
   */
  inline auto getBitOffset() const {
    return criterion.getBitOffset();
  }


  /**
     @brief Advances to next node when observations are all numerical.

     @param rowT is a row base within the transposed numerical set.

     @param[out] leafIdx outputs predictor index iff at terminal.

     @return delta to next node, if nonterminal, else zero.
   */
  inline IndexT advanceNum(const double numVal) const {
    if (trapUnobserved && isnan(numVal))
      return 0;
    else
      return delInvert(invert ? (numVal > getSplitNum()) : (numVal <= getSplitNum()));
  }


  /**
     @brief Advances according to a factor-based criterion.

     Factor criteria are randomized during training, so inversion state may be
     ignored.

     @return delta to branch target.
   */
  inline IndexT advanceFactor(const BV& bits,
			      const BV& bitsObserved,
			      size_t bitOffset) const {
    if (trapUnobserved && !bitsObserved.testBit(bitOffset))
      return 0;
    else
      return delTest(bits.testBit(bitOffset));
  }
  

  /**
     @brief Node advancer for all-factor observations.

     Splitting for factor values is a set-membership relation.  Randomization
     is implemented at training, making it unnecessary to read the inversion
     sense.

     @param rowT holds the transposed factor-valued observations.

     @param tIdx is the tree index.

     @param leafIdx as above.

     @return terminal/nonterminal : 0 / delta to next node.
   */
  IndexT advanceFactor(const vector<BV>& factorBits,
		       const vector<BV>& bitsObserved,
		       const CtgT rowFT[],
		       unsigned int tIdx) const {
    return advanceFactor(factorBits[tIdx], bitsObserved[tIdx], getBitOffset() + rowFT[getPredIdx()]);
  } // EXIT


  IndexT advanceFactor(const BV& factorBits,
		       const BV& bitsObserved,
		       const CtgT rowFT[]) const {
    return advanceFactor(factorBits, bitsObserved, getBitOffset() + rowFT[getPredIdx()]);
  }


  /**
     @brief Node advancer, as above, but for mixed observation.

     Splitting for factor values is a set-membership relation.  Randomization
     is implemented at training, making it unnecessary to read the inversion
     sense.

     Parameters as above, along with:

     @param rowNT contains the transponsed numerical observations.

     @return terminal/nonterminal : 0 / delta to next node.
   */
  IndexT advanceMixed(const class PredictFrame& frame,
		      const vector<class BV>& factorBits,
		      const vector<class BV>& bitsObserved,
		      const CtgT* rowFT,
		      const double *rowNT,
		      unsigned int tIdx) const; // EXIT


  IndexT advanceMixed(const class PredictFrame& frame,
		      const class BV& factorBits,
		      const class BV& bitsObserved,
		      const CtgT* rowFT,
		      const double *rowNT) const;

  /**
     @brief Interplates split values from fractional intermediate rank.

     @param summaryFrame identifies numeric-valued predictors.
   */
  void setQuantRank(const class PredictorFrame* frame);

  
  inline bool getLeafIdx(IndexT& leafIdx) const {
    IndexT delIdx = getDelIdx();
    if (delIdx == 0) {
      leafIdx = criterion.getLeafIdx();
    }
    return delIdx == 0;
  }


  /**
     @brief As above, but assumes node is noterminal.
   */
  inline IndexT getLeafIdx() const {
    return criterion.getLeafIdx();
  }
  

  /**
     @brief Resets delIdx to terminal, preserving remaining state.
   */
  inline void resetTerminal() {
    resetDelIdx(0);
  }
  

  /**
     @brief Sets existing node to leaf state.

     @param leafIdx is the tree-relative leaf index, if tracked.
   */
  inline void setLeaf(IndexT leafIdx) {
    resetDelIdx(0);
    criterion.setLeafIdx(leafIdx);
  }
};


#endif
