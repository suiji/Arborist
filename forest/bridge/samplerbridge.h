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

  /**
     @brief Sampling constructor.
   */
  SamplerBridge(size_t nObs,
		size_t nSamp,
		unsigned int nTree,
		bool replace,
		const double weight[]);

  SamplerBridge(SamplerBridge&& sb);


  /**
     @param bagging specifies a bagging matrix:  prediction only.
   */
  SamplerBridge(const vector<double>& yTrain,
		size_t nSamp,
		unsigned int nTree,
		const double samples[],
		bool bagging = false);


  SamplerBridge(const vector<unsigned int>& yTrain,
		size_t nSamp,
		unsigned int nTree,
		const double samples[],
		unsigned int nCtg,
		const vector<double>& classWeight);


  SamplerBridge(const vector<unsigned int>& yTrain,
		unsigned int nCtg,
		size_t nSamp,
		unsigned int nTree,
		const double samples[],
		bool bagging = false);


  /**
     @brief Generic constructor.
   */
  SamplerBridge(size_t nObs,
		const double samples[],
		size_t nSamp,
		unsigned int nTree);

  
  ~SamplerBridge();


  /**
     @brief Invokes core sampling for a single tree.
   */
  void sample();


  /**
     @brief Gets core Sampler.  Non-constant for training.

     @return core sampler.
    */
  class Sampler* getSampler() const;

  /**
     @brief Getter for number of training rows.
   */
  size_t getNObs() const;


  size_t getNSamp() const;
  
  
  /**
     @brief Getter for number of trained trees.
   */
  unsigned int getNRep() const;


  size_t getNuxCount() const;

  
  /**
     @brief Copies the sampling records into the buffer passed.
   */
  void dumpNux(double nuxOut[]) const;

  
private:

  unique_ptr<class Sampler> sampler; // Core-level instantiation.
};


#endif
