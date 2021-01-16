// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file leafbridge.h

   @brief Front-end wrappers for core Leaf objects.

   @author Mark Seligman
 */

#ifndef TREE_BRIDGE_LEAFBRIDGE_H
#define TREE_BRIDGE_LEAFBRIDGE_H

#include "bagbridge.h"

struct LeafBridge {
  /**
     @brief Getter for number of rows under prediction.
   */
  size_t getRowPredict() const;

  /**
     @brief Constructor for regression prediction.
   */
  LeafBridge(const vector<size_t>& height,
	     const unsigned char* node,
	     const vector<size_t>& bagHeight,
	     const unsigned char* bagSample);

  
  ~LeafBridge();


  void dump(vector<vector<size_t> >& rowTree,
            vector<vector<unsigned int> >& sCountTree,
            vector<vector<double> >& scoreTree,
            vector<vector<unsigned int> >& extentTree,
	    const struct BagBridge& bag = BagBridge()) const;

  class LeafPredict* getLeaf();

private:
  unique_ptr<class LeafPredict> leaf;
};

#endif
