// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file fbcresc.h

   @brief Data structures and methods for growing factor-valued tree blocks.

   @author Mark Seligman
 */

#ifndef TREE_FBCRESC_H
#define TREE_FBCRESC_H

#include <vector>

#include "typeparam.h"


/**
   @brief Manages the crescent factor blocks.
 */
class FBCresc {
  vector<PredictorT> fac;  // Factor-encoding bit vector.
  vector<size_t> height; // Cumulative vector heights, per tree.

public:

  FBCresc(unsigned int treeChunk);

  /**
     @brief Sets the height of the current tree, storage now being known.

     @param tIdx is the tree index.
   */
  void treeCap(unsigned int tIdx);

  /**
     @brief Consumes factor bit vector and notes height.

     @param splitBits is the bit vector.

     @param bitEnd is the final bit position referenced.

     @param tIdx is the current tree index.
   */
  void appendBits(const class BV* splitBIts,
                  size_t bitEnd,
                  unsigned int tIdx);

  /**
     @brief Computes unit size for cross-compatibility of serialization.
   */
  static constexpr size_t unitSize() {
    return sizeof(unsigned int);
  }
  
  /**
     @brief Dumps factor bits as raw data.

     @param[out] facRaw outputs the raw factor data.
   */
  void dumpRaw(unsigned char facRaw[]) const;
  
  /**
     @brief Accessor for the per-tree height vector.

     @return reference to height vector.
   */
  const vector<size_t>& getHeight() const {
    return height;
  }
};

#endif
