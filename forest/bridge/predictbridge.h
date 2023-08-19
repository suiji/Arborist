// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file predictbridge.h

   @brief Bridge data structures and methods for prediction.

   @author Mark Seligman
 */

#ifndef FOREST_BRIDGE_PREDICTBRIDGE_H
#define FOREST_BRIDGE_PREDICTBRIDGE_H


#include <vector>
#include <memory>

using namespace std;

#include "forestbridge.h"
#include "leafbridge.h"
#include "samplerbridge.h"


/**
   @brief Consolidates common components required by all prediction entries.

   These are typically unwrapped by the front end from several data structures.
 */
struct PredictBridge {
  /**
     @brief Constructor boxes training and output summaries.

     @param nThread is the number of OMP threads requested.

     Remaining parameters mirror similarly-named members.
   */
  PredictBridge(unique_ptr<struct RLEFrame> rleFrame_,
                ForestBridge forest_,
		unsigned int nPermute,
		unsigned int nThread);

  virtual ~PredictBridge();

  
  size_t getNRow() const;


  unsigned int getNTree() const;
  

  bool permutes() const;

  
  /**
     @brief Computes Meinshausen-style weight vectors over a set of observations.
     
     @return vector of normalized weight vectors.
   */
  static vector<double> forestWeight(const ForestBridge& forestBridge,
					     const SamplerBridge& samplerBridge,
					     const LeafBridge& leafBridge,
					     const double indices[],
					     size_t nObs,
					     unsigned int nThread);

  
protected:
  unique_ptr<struct RLEFrame> rleFrame; ///< Local ownership
  ForestBridge forestBridge; ///< Local ownership.
  const unsigned int nPermute; ///< # times to permute.
};


struct PredictRegBridge : public PredictBridge {
  PredictRegBridge(unique_ptr<struct RLEFrame> rleFrame_,
		   ForestBridge forestBridge_,
		   SamplerBridge samplerBridge_,
		   LeafBridge leafBridge_,
		   //		   const pair<double, double>& scoreDesc,
		   vector<double> yTest,
		   unsigned int nPermute_,
		   bool indexing,
		   bool trapUnobserved,
		   unsigned int nThread,
		   vector<double> quantile_);


  ~PredictRegBridge();


  /**
     @brief External entry for prediction.

     May be parametrized for separate entry in distributed setting.
   */
  void predict() const;


  /**
     @return reference to cached index vector.
   */
  const vector<size_t>& getIndices() const;
  

  double getSAE() const;


  double getSSE() const;


  const vector<double>& getSSEPermuted() const;

  
  const vector<double>& getYTest() const;
  

  const vector<double>& getYPred() const;
  
  
  /**
     @return vector of predection quantiles iff quant non-null else empty.
   */
  const vector<double>& getQPred() const;

  /**
     @return vector of estimate quantiles iff quant non-null else empty.
   */
  const vector<double>& getQEst() const;

private:
  SamplerBridge samplerBridge; ///< Local ownership.
  LeafBridge leafBridge; ///< " "
  unique_ptr<class PredictReg> predictRegCore;
};


struct PredictCtgBridge : public PredictBridge {
  PredictCtgBridge(unique_ptr<struct RLEFrame> rleFrame_,
		   ForestBridge forestBridge_,
		   SamplerBridge samplerBridge_,
		   LeafBridge leafBridge_,
		   //		   const pair<double, double>& scoreDesc,
		   vector<unsigned int> yTest,
		   unsigned int nPermute_,
		   bool doProb,
		   bool indexing,
		   bool trapUnobserved,
		   unsigned int nThread);
  
  ~PredictCtgBridge();


  /**
     @return reference to cached index vector.
   */
  const vector<size_t>& getIndices() const;
  

  const vector<unsigned int>& getYPred() const;


  const vector<size_t>& getConfusion() const;


  const vector<double>& getMisprediction() const;


  const vector<vector<double>>& getMispredPermuted() const;
  

  double getOOBError() const;


  const vector<double>& getOOBErrorPermuted() const;
  

  /**
     @brief External entry for prediction.

     May be parametrized for separate entry in distributed setting.
   */
  void predict() const;

  unsigned int ctgIdx(unsigned int ctgTest,
                      unsigned int ctgPred) const;
  

  const vector<unsigned int>& getCensus() const;
  

  const vector<double>& getProb() const;
  

private:
  SamplerBridge samplerBridge; // Local ownership.
  LeafBridge leafBridge; // " "
  unique_ptr<class PredictCtg> predictCtgCore;
};


#endif
