// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file leafbridge.h

   @brief Front-end wrappers for training core Leaf objects.

   @author Mark Seligman
 */

#ifndef FOREST_BRIDGE_LEAFBRIDGE_H
#define FOREST_BRIDGE_LEAFBRIDGE_H

#include <memory>

using namespace std;

/**
   @brief Transmits crescent leaf vectors to the front end during training.
 */
struct LeafBridge {

  /**
     @brief Training constructor.
   */
  LeafBridge(const struct SamplerBridge& sb);


  /**
     @brief Default definition.
   */
  ~LeafBridge();


  struct Leaf* getLeaf() const;
  

  /**
     @brief Copies leaf extents as doubles.
   */
  void dumpExtent(double extentOut[]) const;

  

  size_t getExtentSize() const;
  
  /**
     @brief Copies sample indices as doubles.
   */
  void dumpIndex(double indexOut[]) const;


  size_t getIndexSize() const;
  

private:
  
  unique_ptr<struct Leaf> leaf; ///< Core-level instantiation.
};

#endif
