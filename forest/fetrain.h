// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file fetrain.h

   @brief Core handshake with training bridge.

   @author Mark Seligman
 */

#ifndef FOREST_FETRAIN_H
#define FOREST_FETRAIN_H

#include "typeparam.h"

#include <vector>


/**
   @brief Interface class for front end.  Holds simulation-specific parameters
   of the data and constructs forest, leaf and diagnostic structures.
*/
struct FETrain {

  /**
     @brief Booster stat initializer for independent trees.

     @param loss names the loss function.

     @param scorer names the forest scoring function.
   */
  static void initBooster(const string& loss,
			  const string& scorer);


  /**
     @brief Booster state initializer for sequential trees.

     @param loss names the loss function.

     @param scorer names the forest scoring function.

     @param nu is the learning rate.

     @param trackFit is true iff fit reported for each tree.

     @param stopLag specifies # following local fit minimum.
   */
  static void initBooster(const string& loss,
			  const string& scorer,
			  double nu,
			  bool trackFit,
			  unsigned int stopLag);


  /**
     @brief Initializes node scorer.
   */
  static void initNodeScorer(const string& scorer);


  /**
     @brief Registers per-node probabilities of predictor selection.
  */
  static void initProb(unsigned int predFixed,
                       const vector<double> &predProb);

  /**
     @brief Registers tree-shape parameters.
  */
  static void initTree(IndexT leafMax);


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
  static void initMono(const class PredictorFrame* frame,
                       const vector<double> &regMono);


  static void initGrove(bool thinLeaves,
			unsigned int trainBlock);

\
  /**
     @brief Allows DecNode to set its mask widths.
   */
  static void initDecNode(unsigned int nPred);


  /**
     @brief Sets training weights.
   */
  static void initSamples(vector<double> obsWeight);


  /**
     @brief Sets training weights.
   */
  static void initCtg(vector<double> classWeight);

  
  /**
     @brief Copies contents of the score descriptor.
   */
  static void listScoreDesc(double& nu,
			   double& baseScore,
			   string& forestScore);


  /**
     @brief Static de-initializer.
   */
  static void deInit();
};

#endif
