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

#ifndef FOREST_TRAIN_H
#define FOREST_TRAIN_H

#include <string>
#include <vector>

#include "decnode.h" // Algorithm-specific typedef.
#include "samplercresc.h"
#include "forestcresc.h"
#include "pretree.h"


/**
   @brief Interface class for front end.  Holds simulation-specific parameters
   of the data and constructs forest, leaf and diagnostic structures.
*/
class Train {
  static constexpr double slopFactor = 1.2; // Estimates tree growth.
  static unsigned int trainBlock; // Front-end defined buffer size.

  const IndexT nRow; // Number of observations under training.
  const unsigned int treeChunk; // Local number of trees to train.

  unique_ptr<ForestCresc<DecNode>> forest; // Locally-trained forest block.
  vector<double> predInfo; // E.g., Gini gain:  nPred.

  /**
     @brief Trains a chunk of trees.

     @param summaryFrame summarizes the predictor characteristics.
  */
  void trainChunk(const class TrainFrame* frame);

  unique_ptr<class SamplerCresc> sampler; // Crescent sampler.
  
public:

  /**
     @brief Regression constructor.
  */
  Train(const class TrainFrame* frame,
        const vector<double>& y,
        unsigned int treeChunk_);

  
  /**
     @brief Classification constructor.
  */
  Train(const class TrainFrame* frame,
        const vector<unsigned int>& yCtg,
        unsigned int nCtg,
        const vector<double>& yProxy,
        unsigned int nTree,
        unsigned int treeChunk_);

  ~Train();

  
  /**
     @brief Getter for splitting information values.

     @return reference to per-preditor information vector.
   */
  const vector<double> &getPredInfo() const {
    return predInfo;
  }

  static void initBlock(unsigned int trainBlock_);

 
  /**
     @brief Static de-initializer.
   */
  static void deInit();

  static unique_ptr<Train>
  regression(const class TrainFrame* frame,
	     const vector<double>& y,
	     unsigned int treeChunk);


  static unique_ptr<Train>
  classification(const class TrainFrame* frame,
		 const vector<unsigned int>& yCtg,
		 const vector<double>& yProxy,
		 unsigned int nCtg,
		 unsigned int treeChunk,
		 unsigned int nTree);

  /**
     @brief Attempts to extimate storage requirements for block after
     training first tree.
   */
  void reserve(vector<unique_ptr<PreTree>> &treeBlock);

  /**
     @brief Accumulates block size parameters as clues to forest-wide sizes.

     Estimates improve with larger blocks, at the cost of higher memory
     footprint.  Obsolete:  about to exit.

     @return sum of tree sizes over block.
  */
  unsigned int blockPeek(vector<unique_ptr<PreTree>> &treeBlock,
                         size_t& blockFac,
                         IndexT& blockBag,
                         IndexT& blockLeaf,
                         IndexT& maxHeight);

  /**
     @brief Builds segment of decision forest for a block of trees.

     @param treeBlock is a vector of Sample, PreTree pairs.

     @param blockStart is the starting tree index for the block.
  */
  void blockConsume(vector<unique_ptr<PreTree>> &treeBlock,
                    unsigned int blockStart);

  /**
     @brief  Creates a block of root samples and trains each one.

     @return Wrapped collection of Sample, PreTree pairs.
  */
  vector<unique_ptr<PreTree>> blockProduce(const class TrainFrame* frame,
                                unsigned int tStart,
                                unsigned int tCount);

  /**
     @brief Getter for raw forest pointer.
   */
  const ForestCresc<DecNode>* getForest() const {
    return forest.get();
  }


  /**
     @brief Low-level interface to Sampler writer.
   */
  void cacheSamplerRaw(unsigned char *blRaw) const;
  

  /**
     @brief Low-level interface to Sampler height writer.
   */
  const vector<size_t>& getSamplerHeight() const;
};

#endif
