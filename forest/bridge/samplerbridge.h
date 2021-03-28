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
     @brief Regression constructor.
   */
  SamplerBridge(const vector<double>& yTrain,
		unsigned int nTree_,
		const unsigned char* samplerNux);


  /**
     @brief Categorical constructor.
   */
  SamplerBridge(const vector<unsigned int>& yTrain,
		unsigned int nCtg,
		unsigned int nTree,
		const unsigned char* samplerNux);


  /**
     @return core sampler.
    */
  const struct Sampler* getSampler() const;

  /**
     @brief Getter for number of training rows.
   */
  unsigned int getNObs() const;

  /**
     @brief Getter for number of trained trees.
   */
  unsigned int getNTree() const;

private:

  unique_ptr<class Sampler> sampler; // Core-level instantiation.
};


#endif
