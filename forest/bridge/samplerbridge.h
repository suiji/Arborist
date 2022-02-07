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


  static unique_ptr<SamplerBridge> crescReg(const vector<double>& yTrain,
					    unsigned int nSamp,
					    unsigned int treeChunk);


  /**
     @brief Regression factory:  post-training.
   */
  static unique_ptr<SamplerBridge> readReg(const vector<double>& yTrain,
					   unsigned int nSamp,
					   unsigned int nTree,
					   const double samples[],
					   bool bagging);


  SamplerBridge(const vector<double>& yTrain,
		unsigned int nSamp,
		unsigned int treeChunk);


  /**
     @brief Regression constructor:  post-training.
   */
  SamplerBridge(const vector<double>& yTrain,
		unsigned int nSamp,
		vector<vector<class SamplerNux>> samples,
		bool bagging);


  /**
     @brief Classification constructor:  training.
   */
  static unique_ptr<SamplerBridge> crescCtg(const vector<unsigned int>& yTrain,
					    unsigned int nSamp,
					    unsigned int treeChunk,
					    unsigned int nCtg,
					    const vector<double>& classWeight);


  SamplerBridge(const vector<unsigned int>& yTrain,
		unsigned int nSamp,
		unsigned int treeChunk,
		unsigned int nCtg,
		const vector<double>& classWeight);


  /**
     @brief Categorical constructor:  post-training.
   */
  static unique_ptr<SamplerBridge> readCtg(const vector<unsigned int>& yTrain,
					   unsigned int nCtg,
					   unsigned int nSamp,
					   unsigned int nTree,
					   const double samples[],
					   bool bagging);


  SamplerBridge(const vector<unsigned int>& yTrain,
		unsigned int nSamp,
		vector<vector<class SamplerNux>> samples,
		unsigned int nCtg,
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


  size_t getNuxCount() const;

  
  /**
     @brief Copies the sampling records into the buffer passed.
   */
  void dumpNux(double nuxOut[]) const;

  
private:

  unique_ptr<class Sampler> sampler; // Core-level instantiation.
};


#endif
