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

class Sampler;
class Predict;

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
		const vector<double>& weight,
		size_t nHoldout,
		unsigned int nFold,
		const vector<size_t>& undefined);


  SamplerBridge(SamplerBridge&& sb);


  /**
     @brief Training constructor:  classification.
   */
  SamplerBridge(vector<unsigned int> yTrain,
		size_t nSamp,
		unsigned int nTree,
		const double samples[],
		unsigned int nCtg);

  
  /**
     @brief Training constuctor:  regression.
   */
  SamplerBridge(vector<double> yTrain,
		size_t nSamp,
		unsigned int nTree,
		const double samples[]);

  SamplerBridge(vector<double> yTrain,
		size_t nSamp,
		unsigned int nTree,
		const double samples[],
		unique_ptr<struct RLEFrame> rleFrame);



  SamplerBridge(vector<unsigned int> yTrain,
		unsigned int nCtg,
		size_t nSamp,
		unsigned int nTree,
		const double samples[],
		unique_ptr<struct RLEFrame> rleFrame);


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
  Sampler* getSampler() const;


  Predict* getPredict() const;
  

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


  /**
     @return true iff response is categorical.
   */
  bool categorical() const;


  unique_ptr<struct PredictRegBridge> predictReg(struct ForestBridge&,
						 vector<double> yTest) const;

  
  unique_ptr<struct PredictCtgBridge> predictCtg(struct ForestBridge&,
						 vector<unsigned int> yTest) const;

  
  
private:

  unique_ptr<Sampler> sampler; // Core-level instantiation.
};


#endif
