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

#ifndef CORE_BAGBRIDGE_H
#define CORE_BAGBRIDGE_H

#include<memory>
using namespace std;

/**
   @brief Hides class Bag internals from bridge via forward declarations.
 */
struct BagBridge {

  BagBridge(unsigned int nRow_,
            unsigned int nTree_,
            unsigned char* raw);

  /**
     @brief Constructor for empty bit matrix.
   */
  BagBridge();
  
  ~BagBridge();


  class Bag* getBag() const;
  
  /**
     @brief Getter for number of training rows.
   */
  unsigned int getNRow() const;

  /**
     @brief Getter for number of trained trees.
   */
  unsigned int getNTree() const;
  
  
  /**
     @brief Getter for raw data pointer.

     @return raw pointer if non-empty, else nullptr.
   */
  const class BitMatrix* getRaw() const;
private:

  unique_ptr<class Bag> bag; // Core-level instantiation.
};

#endif
