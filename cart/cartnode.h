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


#ifndef CART_CARTNODE_H
#define CART_CARTNODE_H

#include "crit.h"


/**
   @brief To replace parallel array access.
 */
struct CartNode {
\
  /**
     @brief Nodes must be explicitly set to non-terminal (lhDel != 0).
   */
  CartNode() : lhDel(0) {
  }
  
  /**
     @brief Getter for splitting predictor.

     @return splitting predictor index.
   */
  inline PredictorT getPredIdx() const {
    return criterion.predIdx;
  }


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
  inline auto getSplitBit() const {
    return criterion.getBitOffset();
  }
  

  /**
     @brief Advances to next node when observations are all numerical.

     @param rowT is a row base within the transposed numerical set.

     @param[out] leafIdx outputs predictor index iff at terminal.

     @return delta to next node, if nonterminal, else zero.
   */
  inline IndexT advance(const double *rowT,
			IndexT& leafIdx) const {
    auto predIdx = getPredIdx();
    if (lhDel == 0) {
      leafIdx = predIdx;
      return 0;
    }
    else {
      return rowT[predIdx] <= getSplitNum() ? lhDel : lhDel + 1;
    }
  }

  /**
     @brief Node advancer, as above, but for all-categorical observations.

     @param facSplit accesses the per-tree packed factor-splitting vectors.

     @param rowT holds the transposed factor-valued observations.

     @param tIdx is the tree index.

     @param leafIdx as above.

     @return terminal/nonterminal : 0 / delta to next node.
   */
  IndexT advance(const class BVJagged *facSplit,
		 const IndexT* rowT,
		 unsigned int tIdx,
		 IndexT& leafIdx) const;
  
  /**
     @brief Node advancer, as above, but for mixed observation.

     Parameters as above, along with:

     @param rowNT contains the transponsed numerical observations.

     @return terminal/nonterminal : 0 / delta to next node.
   */
  IndexT advance(const class PredictFrame* blockFrame,
		 const BVJagged *facSplit,
		 const IndexT* rowFT,
		 const double *rowNT,
		 unsigned int tIdx,
		 IndexT& leafIdx) const;


  /**
     @brief Interplates split values from fractional intermediate rank.

     @param summaryFrame identifies numeric-valued predictors.
   */
  void setQuantRank(const class SummaryFrame* summaryFrame);

  
  /**
     @brief Copies decision node, converting offset to numeric value.

     @param lhDel is the relative displacement of the left-hand branch.

     @param crit encodes the splitting criterion.
   */
  inline void setBranch(IndexT lhDel,
                        const Crit& crit) {
    this->lhDel = lhDel;
    criterion = crit;
  }


  /**
     @brief Resets as leaf node.

     @param leafIdx is the tree-relative leaf index.
   */
  inline void setLeaf(IndexT leafIdx) {
    lhDel = 0;
    criterion.predIdx = leafIdx;
    criterion.setNum(0.0);
  }


  /**
     @brief Indicates whether node is nonterminal.

     @return True iff lhDel value is nonzero.
   */
  inline bool isNonterminal() const {
    return lhDel != 0;
  }  


  /**
     @brief Getter for lh-delta value.

     @return lhDel value.
   */
  inline auto getLHDel() const {
    return lhDel;
  }

  
private:
  IndexT lhDel;  // Delta to LH subnode. Nonzero iff non-terminal.
  Crit criterion;
};

#endif
