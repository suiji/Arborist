// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file treenode.h

   @brief Data structures and methods implementing tree nodes.

   @author Mark Seligman
 */


#ifndef FOREST_TREENODE_H
#define FOREST_TREENODE_H

#include "typeparam.h"
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

  PackedT packed;

protected:

  Crit criterion;
  bool invert;

public:
  /**
     @brief Initializes packing parameters.
   */
  static void initMasks(PredictorT nPred);


  /**
     @brief Resets packing values to default.
   */
  static void deInit();


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
  TreeNode(complex<double> pair);


  /**
     @brief Encodes node contents as complex value.

     @param[out] valOut outputs the complex encoding.
   */
  void dump(complex<double>& valOut) const {
    valOut = complex<double>(invert ? -double(packed) : packed, criterion.getVal());
  }

  
  /**
     @brief Sets the invert field to the specified (randomized) sense.
   */
  void setInvert(bool invert) {
    this->invert = invert;
  }


  void setPredIdx(PredictorT predIdx) {
    packed |= predIdx;
  }
  

  /**
     @brief Getter for splitting predictor.

     @return splitting predictor index.
   */
  PredictorT getPredIdx() const {
    return packed & rightMask;
  }


  /**
     @brief Initializes delIdx value; not for resetting.
   */
  void setDelIdx(IndexT delIdx) {
    packed |= (size_t(delIdx) << rightBits);
  }
  

  void resetDelIdx(IndexT delIdx) {
    packed = getPredIdx();
    packed |= (size_t(delIdx) << rightBits);
  }
  

  /**
     @brief Getter for lh-delta value.

     @return delIdx value.
   */
  IndexT getDelIdx() const {
    return packed >> rightBits;
  }


  /**
     @brief Indicates whether node is nonterminal.

     @return True iff delIdx value is nonzero.
   */
  bool isNonterminal() const {
    return getDelIdx() != 0;
  }  


  bool isTerminal() const {
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
  auto getSplitNum() const {
    return criterion.getNumVal();
  }


  /**
     @return first bit position of split.
   */
  auto getBitOffset() const {
    return criterion.getBitOffset();
  }


  /**
     @brief Advances to next node when observations are all numerical.

     @param rowT is a row base within the transposed numerical set.

     @param[out] leafIdx outputs predictor index iff at terminal.

     @return delta to next node, if nonterminal, else zero.
   */
  IndexT advanceNum(const double numVal) const {
    return delInvert(invert ? (numVal > getSplitNum()) : (numVal <= getSplitNum()));
  }


  /**
     @brief As above, but traps NaN (unobserved) frame values.
   */
  IndexT advanceNumTrap(const double numVal) const {
    if (isnan(numVal))
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
  IndexT advanceFactor(const BV& bits,
		       size_t bitOffset) const {
    return delTest(bits.testBit(bitOffset));
  }


  /**
     @brief As above, but traps on unobserved bits.
   */
  IndexT advanceFactorTrap(const BV& bits,
			   const BV& bitsObserved,
			   size_t bitOffset) const {
    if (!bitsObserved.testBit(bitOffset))
      return 0;
    else
      return delTest(bits.testBit(bitOffset));
  }


  /**
     @brief Interplates split values from fractional intermediate rank.

     @param summaryFrame identifies numeric-valued predictors.
   */
  void setQuantRank(const class PredictorFrame* frame);

  
  bool getLeafIdx(IndexT& leafIdx) const {
    IndexT delIdx = getDelIdx();
    if (delIdx == 0) {
      leafIdx = criterion.getLeafIdx();
    }
    return delIdx == 0;
  }


  /**
     @brief As above, but assumes node is noterminal.
   */
  IndexT getLeafIdx() const {
    return criterion.getLeafIdx();
  }
  

  /**
     @brief Resets delIdx to terminal, preserving remaining state.
   */
  void resetTerminal() {
    resetDelIdx(0);
  }
  

  /**
     @brief Sets existing node to leaf state.

     @param leafIdx is the tree-relative leaf index, if tracked.
   */
  void setLeaf(IndexT leafIdx) {
    resetDelIdx(0);
    criterion.setLeafIdx(leafIdx);
  }
};


#endif
