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
                unique_ptr<struct ForestBridge> forest_,
		unsigned int nPermute,
		unsigned int nThread);

  virtual ~PredictBridge();

  
  size_t getNRow() const;


  bool permutes() const;


protected:
  unique_ptr<struct RLEFrame> rleFrame; // Local ownership
  unique_ptr<struct ForestBridge> forestBridge; // Local ownership.
  const unsigned int nPermute; // # times to permute.
};


struct PredictRegBridge : public PredictBridge {
  PredictRegBridge(unique_ptr<struct RLEFrame> rleFrame_,
		   unique_ptr<struct ForestBridge> forestBridge_,
		   unique_ptr<struct SamplerBridge> samplerBridge_,
		   unique_ptr<struct LeafBridge> leafBridge_,
		   vector<double> yTest,
		   unsigned int nPermute_,
		   bool trapUnobserved,
		   unsigned int nThread,
		   vector<double> quantile_);

  ~PredictRegBridge(); // Forward declaration:  not specified default.


  /**
     @brief External entry for prediction.

     May be parametrized for separate entry in distributed setting.
   */
  void predict() const;


  double getSAE() const;


  double getSSE() const;


  const vector<double>& getSSEPermuted() const;

  
  const vector<double>& getYTest() const;
  

  const vector<double>& getYPred() const;
  
  
  /**
     @return vector of predection quantiles iff quant non-null else empty.
   */
  const vector<double> getQPred() const;

  /**
     @return vector of estimate quantiles iff quant non-null else empty.
   */
  const vector<double> getQEst() const;
  
private:
  unique_ptr<struct SamplerBridge> samplerBridge; // Local ownership.
  unique_ptr<struct LeafBridge> leafBridge;
  unique_ptr<class PredictReg> predictRegCore;
};


struct PredictCtgBridge : public PredictBridge {
  PredictCtgBridge(unique_ptr<struct RLEFrame> rleFrame_,
		   unique_ptr<struct ForestBridge> forestBridge_,
		   unique_ptr<SamplerBridge> samplerBridge_,
		   unique_ptr<struct LeafBridge> leafBridge_,
		   vector<unsigned int> yTest,
		   unsigned int nPermute_,
		   bool doProb,
		   bool trapUnobserved,
		   unsigned int nThread);

  ~PredictCtgBridge(); // Forward declaration:  not specified default.


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
  

  const unsigned int* getCensus() const;
  

  const vector<double>& getProb() const;
  

private:
  unique_ptr<struct SamplerBridge> samplerBridge; // Local ownership.
  unique_ptr<struct LeafBridge> leafBridge; // " "
  unique_ptr<class PredictCtg> predictCtgCore;


};


#endif
