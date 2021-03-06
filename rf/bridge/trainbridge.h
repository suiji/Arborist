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


#ifndef RF_BRIDGE_TRAINBRIDGE_H
#define RF_BRIDGE_TRAINBRIDGE_H

#include "forestcresc.h"
#include "cartnode.h"

#include<vector>
#include<memory>

using namespace std;

struct TrainBridge {
  TrainBridge(const struct RLEFrame* rleFrame,
	      double autoCompress,
	      bool enableCoproc,
	      vector<string>& diag);
  
  ~TrainBridge();

  /**
     @brief Copies internal-to-external predictor map.

     @return copy of trainFrame's predMap.
   */
  vector<PredictorT> getPredMap() const;
  

  unique_ptr<struct TrainChunk> classification(const vector<unsigned int>& yCtg,
		 const vector<double>& yProxy,
		 unsigned int nCtg,
		 unsigned int treeChunk,
		 unsigned int nTree) const;


  unique_ptr<struct TrainChunk>
  regression(const vector<double>& y,
	     unsigned int treeChunk) const;

  /**
     @brief Registers training tree-block count.

     @param trainBlock_ is the number of trees by which to block.
  */
  static void initBlock(unsigned int trainBlock);


  static void initProb(unsigned int predFixed,
                       const vector<double> &predProb);


  /**
     @brief Registers tree-shape parameters.
  */
  static void initTree(unsigned int nSamp,
                       unsigned int minNode,
                       unsigned int leafMax);

  /**
     @brief Initializes static OMP thread state.

     @param nThread is a user-specified thread request.
   */
  static void initOmp(unsigned int nThread);


  /**
     @brief Registers response-sampling parameters.

     @param nSamp is the number of samples requested.
  */
  static void initSample(unsigned int nSamp);

  /**
     @brief Registers width of categorical response.

     @pram ctgWidth is the number of training response categories.
  */
  static void initCtgWidth(unsigned int ctgWidth);

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
  unique_ptr<class TrainFrame> trainFrame;
};


struct TrainChunk {
  TrainChunk(unique_ptr<class Train>);

  ~TrainChunk();
  

  void writeSamplerBlockHeight(vector<size_t>& samplerHeight,
			  unsigned int tIdx) const;

  /**
     @brief As above, but Sampler.
   */
  bool samplerBlockFits(const vector<size_t>& height,
                     unsigned int tIdx,
                     size_t capacity,
                     size_t& offset,
                     size_t& bytest) const;
  
  /**
     @brief Sends trained Forest components to front end.
   */
  const vector<size_t>& getForestHeight() const;

  const vector<size_t>& getFactorHeight() const;

  void dumpTreeRaw(unsigned char treeOut[]) const;

  void dumpFactorRaw(unsigned char facOut[]) const;


  const vector<size_t>& samplerHeight() const;


  void dumpSamplerBlockRaw(unsigned char blOut[]) const;


  /**
     @brief Getter for raw forest pointer.
   */
  const class ForestCresc<struct TreeNode>* getForest() const;

  
  /**
     @brief Getter for splitting information values.

     @return reference to per-preditor information vector.
   */
  const vector<double>& getPredInfo() const;


private:

    unique_ptr<class Train> train;
};


#endif
