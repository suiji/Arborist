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

     Remaining parameters mirror similarly-named members.
   */
  PredictBridge();

  
  virtual ~PredictBridge();


  static void initPredict(bool indexing,
			  bool bagging,
			  unsigned int nPermute,
			  bool trapUnobserved);
  

  /**
     @brief Initializes quantile reporting.
   */
  static void initQuant(vector<double> quantile);


  /**
     @brief Initializes categorical probability recording.
   */
  static void initCtgProb(bool doProb);


  size_t getNObs() const;


  /**
     @brief Computes Meinshausen-style weight vectors over a set of observations.
     
     @return vector of normalized weight vectors.
   */
  static vector<double> forestWeight(const struct ForestBridge& forestBridge,
				     const struct SamplerBridge& samplerBridge,
				     const double indices[],
				     size_t nObs);
};


struct PredictRegBridge : public PredictBridge {
  PredictRegBridge(unique_ptr<struct SummaryReg> summary_);


  ~PredictRegBridge();


  unique_ptr<struct SummaryReg> summary;

  

  /**
     @brief External entry for prediction.

     May be parametrized for separate entry in distributed setting.
   */
  static unique_ptr<PredictRegBridge> predict(const class Sampler* sampler,
					      class Forest* forest,
				       vector<double> yTest);

  bool permutes() const;


  size_t getNObs() const;


  
  /**
     @return reference to cached index vector.
   */
  const vector<size_t>& getIndices() const;
  

  double getSAE() const;


  double getSSE() const;


  vector<vector<double>> getSSEPermuted() const;

  
  vector<vector<double>> getSAEPermuted() const;

  
  const vector<double>& getYPred() const;
  
  
  /**
     @return vector of predection quantiles iff quant non-null else empty.
   */
  const vector<double>& getQPred() const;

  /**
     @return vector of estimate quantiles iff quant non-null else empty.
   */
  const vector<double>& getQEst() const;
};


struct PredictCtgBridge : public PredictBridge {
  PredictCtgBridge(unique_ptr<struct SummaryCtg> summary);


  ~PredictCtgBridge();


  unique_ptr<struct SummaryCtg> summary;
  

  bool permutes() const;


  size_t getNObs() const;

  
  /**
     @return reference to cached index vector.
   */
  const vector<size_t>& getIndices() const;
  

  const vector<unsigned int>& getYPred() const;


  const vector<size_t>& getConfusion() const;


  const vector<double>& getMisprediction() const;


  vector<vector<vector<double>>> getMispredPermuted() const;
  

  double getOOBError() const;


  vector<vector<double>> getOOBErrorPermuted() const;
  

  /**
     @brief External entry for prediction.

     May be parametrized for separate entry in distributed setting.
   */
  static unique_ptr<PredictCtgBridge> predict(const class Sampler* sampler,
					      class Forest* forest,
					      vector<unsigned int> yTest);


  unsigned int ctgIdx(unsigned int ctgTest,
                      unsigned int ctgPred) const;
  

  const vector<unsigned int>& getCensus() const;
  

  const vector<double>& getProb() const;
};


#endif
