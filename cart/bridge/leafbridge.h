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

#ifndef CORE_BRIDGE_LEAFBRIDGE_H
#define CORE_BRIDGE_LEAFBRIDGE_H

struct LeafBridge {
  /**
     @brief Getter for number of rows under prediction.
   */
  size_t getRowPredict() const;

  /**
     @brief Constructor for regression prediction.
   */
  LeafBridge(const unsigned int* height,
	     unsigned int nTree,
	     const unsigned char* node,
	     const unsigned int* bagHeight,
	     const unsigned char* bagSample);

  
  ~LeafBridge();


  void dump(const struct BagBridge* bagBridge,
            vector<vector<size_t> >& rowTree,
            vector<vector<unsigned int> >& sCountTree,
            vector<vector<double> >& scoreTree,
            vector<vector<unsigned int> >& extentTree) const;

  class LeafPredict* getLeaf();

private:
  unique_ptr<class LeafPredict> leaf;
};

#endif
