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

#ifndef FOREST_BRIDGE_LEAFBRIDGE_H
#define FOREST_BRIDGE_LEAFBRIDGE_H

#include <memory>
#include <vector>

using namespace std;

/**
   @brief Hides class Sampler internals from bridge via forward declarations.
 */
struct LeafBridge {
  static unique_ptr<LeafBridge> FactoryTrain(const struct SamplerBridge* sb,
					     bool thin);


  static unique_ptr<LeafBridge> FactoryPredict(const struct SamplerBridge* samplerBridge,
					       bool thin,
					       const double extent_[],
					       const double index_[]);


  LeafBridge(const struct SamplerBridge* sb,
	     bool thin);
  

  LeafBridge(const struct SamplerBridge* samplerBridge,
	     bool thin,
	     vector<vector<size_t>> extent,
	     vector<vector<vector<size_t>>> index);


  ~LeafBridge();


  struct Leaf* getLeaf() const;
  

  
  static vector<vector<size_t>> unpackExtent(const struct SamplerBridge* samplerBridge,
					     bool thin,
					     const double numVal[]);

  
  static vector<vector<vector<size_t>>> unpackIndex(const struct SamplerBridge* samplerBridge,
						    bool thin,
						    const vector<vector<size_t>>& extent,
						    const double numVal[]);


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
  
  unique_ptr<struct Leaf> leaf;

};

#endif
