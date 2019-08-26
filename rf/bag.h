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

#ifndef CORE_BAG_H
#define CORE_BAG_H

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

     @param oob is true iff out-of-bag sampling is specified.

     @param tIdx is the tree index.

     @param row is the row index.

     @return true iff oob sampling specified and the coordinate bit is set.
   */
  inline bool isBagged(bool oob, unsigned int tIdx, size_t row) const {
    return oob && bitMatrix->testBit(tIdx, row);
  }


  class BitMatrix* getBitMatrix() const;
};


#endif
