// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file trainbridge.h

   @brief Training methods exportable to front end.

   @author Mark Seligman
 */


#ifndef FOREST_BRIDGE_TRAINBRIDGE_H
#define FOREST_BRIDGE_TRAINBRIDGE_H

#include<vector>
#include<memory>

using namespace std;

class PredictorFrame;

struct TrainBridge {
  TrainBridge(unique_ptr<struct RLEFrame> rleFrame,
	      double autoCompress,
	      bool enableCoproc,
	      vector<string>& diag);

  
  ~TrainBridge();


  const PredictorFrame* getFrame() const {
    return frame.get();
  }

  
  /**
     @brief Copies internal-to-external predictor map.

     @return copy of frame's predMap.
   */
  vector<unsigned int> getPredMap() const;

  /**
     @brief Invokes DecNode's static initializer.
   */
  static void init(unsigned int nPred);


  /**
     @brief Registers training parameters for a grove ot trees.

     @param thinLeaves is true iff leaf information elided.
     
     @param trainBlock is the number of trees by which to block.
  */
  static void initGrove(bool thinLeaves,
			unsigned int trainBlock);


  static void initProb(unsigned int predFixed,
                       const vector<double> &predProb);

  /**
     @brief Registers tree-shape parameters.
  */
  static void initTree(size_t leafMax);


  static void initSamples(vector<double> obsWeight);

  
  static void initCtg(vector<double> classWeight);

  
  /**
     @brief Sets loss and scoring for independent forest.
   */
  static void initBooster(const string& loss,
			  const string& scorer);

  
  /**
     @brief Sets update for sequential forest,
   */
  static void initBooster(const string& loss,
			  const string& scorer,
			  double nu,
			  bool trackFit,
			  unsigned int stopLag);


  /**
     @brief Deconstructs contents of core object's ScoreDesc.
   */
  static void getScoreDesc(double& nu,
			   double& baseScore,
			   string& forestScorer);


  static void initNodeScorer(const string& scorer);


  /**
     @brief Initializes static OMP thread state.

     @param nThread is a user-specified thread request.
   */
  static void initOmp(unsigned int nThread);

  
  /**
     @brief Registers parameters governing splitting.
     
     @param minNode is the mininal number of sample indices represented by a tree node.

     @param totLevels is the maximum tree depth to train.

     @param minRatio is the minimum information ratio of a node to its parent.
     
     @param splitQuant is a per-predictor quantile specification.
  */
  static void initSplit(unsigned int minNode,
                        unsigned int totLevels,
                        double minRatio,
			const vector<double>& feSplitQuant);
  
  /**
     @brief Registers monotone specifications for regression.

     @param regMono has length equal to the predictor count.  Only
     numeric predictors may have nonzero entries.
  */
  void initMono(const vector<double>& regMono);


  /**
     @brief Static de-initializer.
   */
  static void deInit();

private:
  unique_ptr<PredictorFrame> frame;
};

#endif
