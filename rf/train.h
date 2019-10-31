// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file train.h

   @brief Class definitions for the training entry point.

   @author Mark Seligman
 */

#ifndef RF_TRAIN_H
#define RF_TRAIN_H

#include <string>
#include <vector>

#include "decnode.h"
#include "forestcresc.h"
#include "pretree.h"


/**
   @brief Short-lived bundle of objects created for training a block of trees.
 */
typedef pair<unique_ptr<class Sample>, unique_ptr<PreTree> > TrainSet;

/**
   @brief Interface class for front end.  Holds simulation-specific parameters
   of the data and constructs forest, leaf and diagnostic structures.
*/
class Train {
  static constexpr double slopFactor = 1.2; // Estimates tree growth.
  static unsigned int trainBlock; // Front-end defined buffer size.

  const unique_ptr<class CandRF> cand; // Pre-candidate choice methods.
  const unsigned int nRow; // Number of rows to train.
  const unsigned int treeChunk; // Local number of trees to train.
  unique_ptr<class BitMatrix> bagRow; // Local bag section:  treeChunk x nRow
  unique_ptr<ForestCresc<DecNode>> forest; // Locally-trained forest block.
  vector<double> predInfo; // E.g., Gini gain:  nPred.

  /**
     @brief Trains a chunk of trees.

     @param summaryFrame summarizes the predictor characteristics.
  */
  void trainChunk(const class SummaryFrame* frame);

  unique_ptr<class LFTrain> leaf; // Crescent leaf object.

public:

  /**
     @brief Regression constructor.
  */
  Train(const class SummaryFrame* frame,
        const double* y,
        unsigned int treeChunk_);

  
  /**
     @brief Classification constructor.
  */
  Train(const class SummaryFrame* frame,
        const unsigned int* yCtg,
        unsigned int nCtg,
        const double* yProxy,
        unsigned int nTree,
        unsigned int treeChunk_);

  ~Train();


  /**
     @brief Getter for raw leaf pointer.
   */
  const class LFTrain *getLeaf() const {
    return leaf.get();
  }


  /**
     @brief Getter for splitting information values.

     @return reference to per-preditor information vector.
   */
  const vector<double> &getPredInfo() const {
    return predInfo;
  }


  /**
     @brief Static initialization methods.
  */

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
  static void initMono(const class SummaryFrame* frame,
                       const vector<double> &regMono);

  /**
     @brief Static de-initializer.
   */
  static void deInit();

  static unique_ptr<Train>
  regression(const class SummaryFrame* frame,
	     const double *y,
	     unsigned int treeChunk);


  static unique_ptr<Train>
  classification(const class SummaryFrame* frame,
		 const unsigned int *yCtg,
		 const double *yProxy,
		 unsigned int nCtg,
		 unsigned int treeChunk,
		 unsigned int nTree);

  /**
     @brief Attempts to extimate storage requirements for block after
     training first tree.
   */
  void reserve(vector<TrainSet> &treeBlock);

  /**
     @brief Accumulates block size parameters as clues to forest-wide sizes.

     Estimates improve with larger blocks, at the cost of higher memory
     footprint.  Obsolete:  about to exit.

     @return sum of tree sizes over block.
  */
  unsigned int blockPeek(vector<TrainSet> &treeBlock,
                         size_t& blockFac,
                         IndexT& blockBag,
                         IndexT& blockLeaf,
                         IndexT& maxHeight);

  /**
     @brief Builds segment of decision forest for a block of trees.

     @param treeBlock is a vector of Sample, PreTree pairs.

     @param blockStart is the starting tree index for the block.
  */
  void blockConsume(vector<TrainSet> &treeBlock,
                    unsigned int blockStart);

  /**
     @brief  Creates a block of root samples and trains each one.

     @return Wrapped collection of Sample, PreTree pairs.
  */
  vector<TrainSet> blockProduce(const class SummaryFrame* frame,
                                unsigned int tStart,
                                unsigned int tCount);

  /**
     @brief Getter for raw forest pointer.
   */
  const ForestCresc<DecNode>* getForest() const {
    return forest.get();
  }

  /**
     @brief Dumps bag contents as raw characters.

     @param[out] bbRaw
   */
  void cacheBagRaw(unsigned char bbRaw[]) const;


  /**
     @brief Fixes splitting regime:  CART, survival, entropy, usw.

     @nCtg is the reponse categoricity.
   */
  unique_ptr<class SplitFrontier>
  splitFactory(const class SummaryFrame* frame,
	       class Frontier* frontier,
	       const class Sample* sample,
	       PredictorT nCtg) const;
};

#endif
