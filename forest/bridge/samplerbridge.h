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

  
  ~SamplerBridge();


  /**
     @brief Sampling entry.
   */
  static unique_ptr<SamplerBridge> preSample(size_t nSamp,
					     size_t nObs,
					     unsigned int nTree,
					     bool replace,
					     const double weight[]);

  /**
     @brief Regression, training entry.
   */
  static unique_ptr<SamplerBridge> trainReg(const vector<double>& yTrain,
					    size_t nSamp,
					    unsigned int nTree,
					    const double samples[]);


  /**
     @brief Regression factory:  post-training.
   */
  static unique_ptr<SamplerBridge> readReg(const vector<double>& yTrain,
					   size_t nSamp,
					   unsigned int nTree,
					   const double samples[],
					   bool bagging);


  SamplerBridge(const vector<double>& yTrain,
		size_t nSamp,
		vector<vector<class SamplerNux>> samples);


  /**
     @brief Regression constructor:  post-training.
   */
  SamplerBridge(const vector<double>& yTrain,
		size_t nSamp,
		vector<vector<class SamplerNux>> samples,
		bool bagging);


  /**
     @brief Classification:  training entry.
   */
  static unique_ptr<SamplerBridge> trainCtg(const vector<unsigned int>& yTrain,
					    size_t nSamp,
					    unsigned int nTree,
					    const double samples[],
					    unsigned int nCtg,
					    const vector<double>& classWeight);


  SamplerBridge(const vector<unsigned int>& yTrain,
		size_t nSamp,
		vector<vector<class SamplerNux>> nux,
		unsigned int nCtg,
		const vector<double>& classWeight);


  /**
     @brief Categorical constructor:  post-training.
   */
  static unique_ptr<SamplerBridge> readCtg(const vector<unsigned int>& yTrain,
					   unsigned int nCtg,
					   size_t nSamp,
					   unsigned int nTree,
					   const double samples[],
					   bool bagging);


  SamplerBridge(const vector<unsigned int>& yTrain,
		size_t nSamp,
		vector<vector<class SamplerNux>> samples,
		unsigned int nCtg,
		bool bagging);

  
  /**
     @brief Invokes core sampling for a single tree.
   */
  void sample();

  
  void appendSamples(const vector<size_t>& idx); // EXIT: internalized.

  
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
