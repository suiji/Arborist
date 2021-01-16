// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file bag.h

   @brief Wrapper for bit-matrix encoding of tree bags.

   @author Mark Seligman
 */

#ifndef OBS_BAG_H
#define OBS_BAG_H

#include "bv.h"
#include "typeparam.h"

class Bag {
  unsigned int nTree;
  size_t nObs;
  unique_ptr<class BitMatrix> bitMatrix;

 public:

  auto getNObs() const {
    return nObs;
  }

  auto getNTree() const {
    return nTree;
  }
  
  Bag(unsigned int* raw_, unsigned int nTree_, size_t nObs_);

  /**
     @brief Constructor for empty bag.
   */
  Bag();


  /**
     @brief Determines whether a given forest coordinate is bagged.

     @param tIdx is the tree index.

     @param row is the row index.

     @return true iff matrix is nonempty and the coordinate bit is set.
   */
  inline bool isBagged(unsigned int tIdx, size_t row) const {
    return nTree != 0 && bitMatrix->testBit(tIdx, row);
  }

  
  bool isEmpty() const {
    return bitMatrix->isEmpty();
  }
  

  class BitMatrix* getBitMatrix() const;
};


#endif
