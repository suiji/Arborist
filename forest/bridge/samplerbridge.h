// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file samplerbridge.h

   @brief Front-end wrappers for core Sampler objects.

   @author Mark Seligman
 */

#ifndef FOREST_BRIDGE_SAMPLERBRIDGE_H
#define FOREST_BRIDGE_SAMPLERBRIDGE_H

#include "typeparam.h"

#include <memory>
#include <vector>

using namespace std;

/**
   @brief Hides class Sampler internals from bridge via forward declarations.
 */
struct SamplerBridge {

  ~SamplerBridge();


  /**
     @brief Computes stride size subsumed by a given observation count.

     @return stride size, in bytes.
   */
  static size_t strideBytes(size_t nObs);


  /**
     @brief Regression constructor:  training.
   */
  SamplerBridge(const vector<double>& yTrain,
		IndexT nSamp,
		unsigned int treeChunk,
		bool thin);


  /**
     @brief Regression constructor:  post-training.
   */
  SamplerBridge(const vector<double>& yTrain,
		IndexT nSamp,
		unsigned int nTree_,
		bool nux,
		unsigned char* samples,
		const double extent[],
		const double index[],
		bool bagging);


  /**
     @brief Classification constructor:  training.
   */
  SamplerBridge(const vector<PredictorT>& yTrain,
		IndexT nSamp,
		unsigned int treeChunk,
		bool thin,
		PredictorT nCtg,
		const vector<double>& classWeight);


  /**
     @brief Categorical constructor:  post-training.
   */
  SamplerBridge(const vector<unsigned int>& yTrain,
		unsigned int nCtg,
		IndexT nSamp,
		unsigned int nTree,
		bool nux,
		unsigned char* samples,
		const double extent[],
		const double index[],
		bool bagging);


  /**
     @brief Gets core Sampler.  Non-constant for training.

     @return core sampler.
    */
  struct Sampler* getSampler() const;

  /**
     @brief Getter for number of training rows.
   */
  unsigned int getNObs() const;

  /**
     @brief Getter for number of trained trees.
   */
  unsigned int getNTree() const;


  size_t getBlockBytes() const;

  
  /**
     @brief Copies the sampling records to the buffer passed.
   */
  void dumpRaw(unsigned char blOut[]) const;

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

  unique_ptr<class Sampler> sampler; // Core-level instantiation.

};


#endif
