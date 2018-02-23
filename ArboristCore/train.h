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

#ifndef ARBORIST_TRAIN_H
#define ARBORIST_TRAIN_H

#include <string>
#include <vector>

#include "typeparam.h"


/**
   @brief Interface class for front end.  Holds simulation-specific parameters
   of the data and constructs forest, leaf and diagnostic structures.
*/
class Train {
  static constexpr double slopFactor = 1.2; // Estimates tree growth.
  static unsigned int trainBlock; // Front-end defined buffer size.

  const unsigned int nTree;
  class ForestTrain *forest;
  vector<double> predInfo; // E.g., Gini gain:  nPred.
  const class Response *response;

  static void DeImmutables();

  void TrainForest(const class FrameTrain *frameTrain,
		   const class RowRank *rowRank);

 public:


  virtual class LeafTrain *Leaf() const = 0;
  

  /**
   @brief Regression constructor.
 */
  Train(const class FrameTrain *frameTrain,
	const double *_y,
	const unsigned int *_row2Rank,
	unsigned int _nTree);

  
  /**
   @brief Classification constructor.
 */
  Train(const class FrameTrain *frameTrain,
	const unsigned int *_yCtg,
	const double *_yProxy,
	unsigned int _nTree);

  virtual ~Train();

  ForestTrain *Forest() const {
    return forest;
  }

  const vector<double> &PredInfo() const {
    return predInfo;
  }
  
  /**
   @brief Static initializer.

   @return void.
 */
  static void InitBlock(unsigned int trainBlock);

  static void InitLeaf(bool thinLeaves);

  static void InitCDF(const vector<double> &splitQuant);

  static void InitProb(unsigned int predFixed,
		       const vector<double> &predProb);


  static void InitTree(unsigned int nSamp,
		       unsigned int minNode,
		       unsigned int leafMax);
  
  static void InitSample(unsigned int nSamp,
			 const vector<double> &sampleWeight,
			 bool withRepl);

  static void InitCtgWidth(unsigned int ctgWidth);

  static void InitSplit(unsigned int minNode,
			unsigned int totLevels,
			double minRatio);
  
  static void InitMono(const vector<double> &regMono);

  static class TrainReg *Regression(const class FrameTrain *frameTrain,
				    const class RowRank *_rowRank,
				    const double *_y,
				    const unsigned int *_row2Rank,
				    unsigned int nTree);

  static class TrainCtg *Classification(const class FrameTrain *frameTrain,
					const class RowRank *_rowRank,
					const unsigned int *_yCtg,
					const double *_yProxy,
					unsigned int nCtg,
					unsigned int nTree);

  void Reserve(vector<class PreTree*> &ptBlock);
  unsigned int BlockPeek(vector<class PreTree*> &ptBlock,
			 unsigned int &blockFac,
			 unsigned int &blockBag,
			 unsigned int &blockLeaf,
			 unsigned int &maxHeight);
  void BlockConsume(const class FrameTrain *frameTrain,
		    const vector<class Sample*> &sampleBlock,
		    vector<class PreTree*> &ptBlock,
		    unsigned int blockStart);
  void TreeBlock(const class FrameTrain *frameTrain,
		 const class RowRank *rowRank,
		 unsigned int tStart,
		 unsigned int tCount);
};


class TrainCtg : public Train {
  class LeafTrainCtg *leafCtg;
 public:

  /**
  */
  TrainCtg(const class FrameTrain *frameTrain,
	   const unsigned int *_yCtg,
	   const double *_yProxy,
	   unsigned int nCtg,
	   unsigned int _nTree);

  ~TrainCtg() {}
  
  class LeafTrainCtg *SubLeaf() const {
    return leafCtg;
  }

  class LeafTrain *Leaf() const;
};


class TrainReg : public Train {
  class LeafTrainReg *leafReg;

 public:
 /**
  */
 TrainReg(const class FrameTrain *frameTrain,
	  const double *_y,
	  const unsigned int *_row2Rank,
	  unsigned int _nTree);

 ~TrainReg() {}
 
  class LeafTrainReg *SubLeaf() const {
    return leafReg;
  }

  class LeafTrain *Leaf() const;
};


#endif
