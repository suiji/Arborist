// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file ptnode.h

   @brief Decision node definitions, characterized by client algorithm.

   @author Mark Seligman

 */

#ifndef TREE_PTNODE_H
#define TREE_PTNODE_H

#include "typeparam.h"
#include "forestcresc.h"
#include "crit.h"

#include <vector>
#include <algorithm>

/**
  @brief Decision node specialized for training.
 */
template<typename nodeType>
class PTNode {
  IndexT lhDel; // zero iff terminal.
  IndexT critCount; // Number of associated criteria; 0 iff terminal.
  IndexT critOffset;  // Index of first criterion.
  FltVal info;  // Zero iff terminal.
 public:

  PTNode() : lhDel(0), critCount(0), info(0.0) { // defaults to terminal.
  }


  inline void bumpCriterion() {
    critCount++;
  }

  /**
     @return starting bit of split value.
   */
  IndexT getBitOffset(const vector<Crit>& crit) const {
    return crit[critOffset].getBitOffset();
  }
  
  
  /**
     @brief Consumes the node fields of nonterminals (splits).

     @param forest[in, out] accumulates the growing forest node vector.
  */
  void consumeNonterminal(ForestCresc<nodeType>* forest,
                          vector<double>& predInfo,
                          IndexT idx,
                          const vector<Crit>& crit) const {
    if (isNonTerminal()) {
      Crit criterion(crit[critOffset]);
      forest->nonTerminal(idx, lhDel, criterion);
      predInfo[criterion.predIdx] += info;
    }
  }



  /**
     @builds bit-based split.

     @param argMax characterizes the split.

     @param lhDel is the distance to the lh-descendant.

     @param critOffset is the begining criterion offset.
   */
  inline void nonterminal(double info,
                          IndexT lhDel,
                          IndexT critOffset) {
    this->info = info;
    this->lhDel = lhDel;
    this->critOffset = critOffset;
  }
  

  /**
     @brief Resets to default terminal status.
   */
  inline void setTerminal() {
    lhDel = 0;
  }


  /**
     @brief Resets to nonterminal with specified lh-delta.

     @return void.
   */
  inline void setNonterminal(IndexT lhDel) {
    this->lhDel = lhDel;
  }

  
  inline bool isNonTerminal() const {
    return lhDel != 0;
  }


  inline IndexT getLHId(IndexT ptId) const {
    return isNonTerminal() ? ptId + lhDel : 0;
  }

  inline IndexT getRHId(IndexT ptId) const {
    return isNonTerminal() ? getLHId(ptId) + 1 : 0;
  }
};

#endif
