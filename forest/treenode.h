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

#include <vector>
#include <complex>

/**
   @brief To replace parallel array access.

   Caches splitting predictor and delta to an explicit branch target.
   Target of complementary branch is computable.  Sense of explicit
   branch and method of computing complement varies with algorithm.
 */
class TreeNode {
  static unsigned int rightBits;
  static PredictorT rightMask;
  PackedT packed;

protected:

  Crit criterion;


public:

  /**
     @brief Initializes packing parameters.
   */
  static void init(PredictorT nPred);

  
  /**
     @brief Resets packing values to default.
   */
  static void deInit();
  

  /**
     @brief Nodes must be explicitly set to non-terminal (delIdx != 0).
   */
  TreeNode() : packed(0ull),
	       criterion(0.0) {
  }


  /**
     @brief Constructor for reading numeric-valued encodings.
   */
  TreeNode(complex<double> pair) :
    packed(pair.real()),
    criterion(pair.imag()) {
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


  inline void setDelIdx(IndexT delIdx) {
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


  void critCut(const class SplitNux* nux,
	       const class SplitFrontier* splitFrontier);


  void critBits(const class SplitNux* nux,
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
     @brief Dumps both members as context-independent values.
   */
  inline void dump(complex<double>& valOut) const {
    valOut = complex<double>(packed, criterion.getVal());
  }

  
  /**
     @brief Advances to next node when observations are all numerical.

     @param rowT is a row base within the transposed numerical set.

     @param[out] leafIdx outputs predictor index iff at terminal.

     @return delta to next node, if nonterminal, else zero.
   */
  inline IndexT advance(const double *rowT) const {
    IndexT delIdx = getDelIdx();
    return delIdx == 0 ? 0 : (delIdx + (rowT[getPredIdx()] <= getSplitNum() ? 0 : 1));
  }

  
  /**
     @brief Node advancer, as above, but for all-categorical observations.

     @param rowT holds the transposed factor-valued observations.

     @param tIdx is the tree index.

     @param leafIdx as above.

     @return terminal/nonterminal : 0 / delta to next node.
   */
  IndexT advance(const vector<unique_ptr<class BV>>& factorBits,
		 const IndexT* rowT,
		 unsigned int tIdx) const;

  
  /**
     @brief Node advancer, as above, but for mixed observation.

     Parameters as above, along with:

     @param rowNT contains the transponsed numerical observations.

     @return terminal/nonterminal : 0 / delta to next node.
   */
  IndexT advance(const class Predict* predict,
		 const vector<unique_ptr<class BV>>& factorBits,
		 const IndexT* rowFT,
		 const double *rowNT,
		 unsigned int tIdx) const;

  /**
     @brief Interplates split values from fractional intermediate rank.

     @param summaryFrame identifies numeric-valued predictors.
   */
  void setQuantRank(const class TrainFrame* trainFrame);

  
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
  

  inline void setTerminal() {
    setDelIdx(0);
  }
  

  /**
     @brief Sets existing node to leaf state.

     @param leafIdx is the tree-relative leaf index, if tracked.
   */
  inline void setLeaf(IndexT leafIdx) {
    setDelIdx(0);
    criterion.setLeafIdx(leafIdx);
  }
};


#endif
