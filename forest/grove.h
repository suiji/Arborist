// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file grove.h

   @brief Trains a block of trees.

   @author Mark Seligman
 */

#ifndef FOREST_GROVE_H
#define FOREST_GROVE_H

#include <string>
#include <vector>

#include "typeparam.h"


/**
   @brief Interface class for front end.  Holds simulation-specific parameters
   of the data and constructs forest, leaf and diagnostic structures.
*/
class Grove {
  static unsigned int trainBlock; ///< Front-end defined buffer size. Unused.

  vector<double> predInfo; ///< E.g., Gini gain:  nPred.
  class Forest* forest; ///< Crescent-state forest block.
  const unique_ptr<struct NodeScorer> nodeScorer; ///< Belongs elsewhere.
  
public:

  /**
     @brief General constructor.
  */
  Grove(const class PredictorFrame* frame,
	const class Sampler* sampler,
	class Forest* forest_,
	unique_ptr<NodeScorer> nodeScorer_);


  void train(const class PredictorFrame* frame,
	     const class Sampler* sampler,
	     const IndexRange& treeRange,
	     struct Leaf* leaf);


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
  static unique_ptr<Grove> trainGrove(const class PredictorFrame* frame,
				 const class Sampler* sampler,
				 class Forest* forest_,
				 const IndexRange& treeRange,
				 struct Leaf* leaf);


  /**
     @brief Builds segment of decision forest for a block of trees.

     @param treeBlock is a vector of Sample, PreTree pairs.
  */
  void blockConsume(const vector<unique_ptr<class PreTree>> &treeBlock,
		    struct Leaf* leaf);


  /**
     @brief  Creates a block of root samples and trains each one.

     @return Wrapped collection of Sample, PreTree pairs.
  */
  vector<unique_ptr<class PreTree>> blockProduce(const class PredictorFrame* frame,
					   const class Sampler* sampler,
					   unsigned int treeStart,
					   unsigned int treeEnd);

  /**
     @brief Accumulates per-predictor information values from trained tree.
   */
  void consumeInfo(const vector<double>& info);


  struct NodeScorer* getNodeScorer() const {
    return nodeScorer.get();
  }
  
  
  /**
     @brief Getter for raw forest pointer.
   */
  const Forest* getForest() const {
    return forest;
  }
};

#endif
