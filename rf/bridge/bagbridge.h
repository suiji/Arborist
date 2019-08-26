// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file bagbridge.h

   @brief Front-end wrappers for core Bag objects.

   @author Mark Seligman
 */

#ifndef RF_BAGBRIDGE_H
#define RF_BAGBRIDGE_H

#include <memory>
using namespace std;

/**
   @brief Hides class Bag internals from bridge via forward declarations.
 */
struct BagBridge {

  BagBridge(size_t nObs,
            unsigned int nTree_,
            unsigned char* raw);

  /**
     @brief Constructor for empty bit matrix.
   */
  BagBridge();
  
  ~BagBridge();


  /**
     @return core bag.
   */
  const class Bag* getBag() const;
  
  
  /**
     @brief Computes stride size subsumed by a given observation count.

     @return stride size, in bytes.
   */
  static size_t strideBytes(size_t nObs);
  

  /**
     @brief Getter for number of training rows.
   */
  unsigned int getNObs() const;

  /**
     @brief Getter for number of trained trees.
   */
  unsigned int getNTree() const;
  

private:

  unique_ptr<class Bag> bag; // Core-level instantiation.
};

#endif
