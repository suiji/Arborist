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
#include "sampler.h"
#include "forest.h"
#include "pretree.h"


/**
   @brief Interface class for front end.  Holds simulation-specific parameters
   of the data and constructs forest, leaf and diagnostic structures.
*/
class Train {
  static unsigned int trainBlock; // Front-end defined buffer size. Unused.

  vector<double> predInfo; // E.g., Gini gain:  nPred.
  class Forest* forest; // Crescent-state forest block.
  class Sampler* sampler; // Crescent-state sampler.


  /**
     @brief Trains a chunk of trees.

     @param frame summarizes the predictor characteristics.

     @param treeChunk is the number of trees in the chunk.
  */
  void trainChunk(const class TrainFrame* frame);
  
public:

  /**
     @brief Regression constructor.
  */
  Train(const class TrainFrame* frame,
	class Forest* forest_,
	class Sampler* sampler_);
  
  
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


  /**
     @brief Main entry to training.
   */
  static unique_ptr<Train> train(const class TrainFrame* frame,
				 class Forest* forest_,
				 class Sampler* sampler_);


  /**
     @brief Builds segment of decision forest for a block of trees.

     @param treeBlock is a vector of Sample, PreTree pairs.
  */
  void blockConsume(const vector<unique_ptr<PreTree>> &treeBlock);


  /**
     @brief  Creates a block of root samples and trains each one.

     @return Wrapped collection of Sample, PreTree pairs.
  */
  vector<unique_ptr<PreTree>> blockProduce(const class TrainFrame* frame,
					   unsigned int treeStart,
					   unsigned int treeEnd) const;

  /**
     @brief Accumulates per-predictor information values from trained tree.
   */
  void consumeInfo(const vector<double>& info);

  
  /**
     @brief Getter for raw forest pointer.
   */
  const Forest* getForest() const {
    return forest;
  }
};

#endif
