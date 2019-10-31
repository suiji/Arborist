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

  unique_ptr<struct TrainChunk>
  classification(const unsigned int *yCtg,
		 const double *yProxy,
		 unsigned int nCtg,
		 unsigned int treeChunk,
		 unsigned int nTree) const;


  unique_ptr<struct TrainChunk>
  regression(const double *y,
	     unsigned int treeChunk) const;

  /**
     @brief Registers training tree-block count.

     @param trainBlock_ is the number of trees by which to block.
  */
  static void initBlock(unsigned int trainBlock);


  /**
     @brief Registers per-node probabilities of predictor selection.
  */
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
  void initMono(const vector<double> &regMono);

  /**
     @brief Static de-initializer.
   */
  static void deInit();

private:
  unique_ptr<class SummaryFrame> summaryFrame;
};


struct TrainChunk {
  TrainChunk(unique_ptr<class Train>);

  ~TrainChunk();
  

  void writeHeight(unsigned int height[],
                   unsigned int tIdx) const;


  void writeBagHeight(unsigned int bagHeight[],
                      unsigned int tIdx) const;


  /**
     @brief Determines whether buffer size is sufficient to accommodate Leaf.

     @param capacity is the current buffer capacity.

     @param[out] offset is the offset of the upcoming write.

     @param[out] bytes is the number of bytes in the upcoming write.

     @return true iff upcoming write fits within current capacity.
   */
  bool leafFits(unsigned int height[],
                unsigned int tIdx,
                size_t capacity,
                size_t& offset,
                size_t& bytest) const;

  /**
     @brief As above, but BagSample.
   */
  bool bagSampleFits(unsigned int height[],
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


  /**
     @brief Sends trained Leaf components to front end.
   */
  const vector<size_t>& getLeafHeight() const;

  void dumpLeafRaw(unsigned char leafOut[]) const;

  const vector<size_t>& leafBagHeight() const;

  void dumpBagLeafRaw(unsigned char blOut[]) const;

  size_t getWeightSize() const;

  void dumpLeafWeight(double weightOut[]) const;


  /**
     @brief Getter for raw leaf pointer.
   */
  const class LFTrain *getLeaf() const;


  /**
     @brief Getter for raw forest pointer.
   */
  const class ForestCresc<struct CartNode>* getForest() const;

  
  /**
     @brief Getter for splitting information values.

     @return reference to per-preditor information vector.
   */
  const vector<double> &getPredInfo() const;

  void dumpBagRaw(unsigned char bbRaw[]) const;

private:

    unique_ptr<class Train> train;
};


#endif
