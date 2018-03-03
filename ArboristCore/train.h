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
  const unique_ptr<class ForestTrain> forest;
  vector<double> predInfo; // E.g., Gini gain:  nPred.
  const unique_ptr<class Response> response;

  void TrainForest(const class FrameTrain *frameTrain,
		   const class RankedSet *rankedPair);

 public:


  virtual class LeafTrain *Leaf() const = 0;
  

  /**
   @brief Regression constructor.
 */
  Train(const class FrameTrain *frameTrain,
	const double *y,
	const unsigned int *row2Rank,
	unsigned int _nTree);

  
  /**
   @brief Classification constructor.
 */
  Train(const class FrameTrain *frameTrain,
	const unsigned int *yCtg,
	const double *yProxy,
	unsigned int _nTree);

  virtual ~Train();

  ForestTrain *Forest() const {
    return forest.get();
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

  static void DeInit();

  static unique_ptr<class TrainReg> Regression(
       const class FrameTrain *frameTrain,
       const class RankedSet *rankedPair,
       const double *y,
       const unsigned int *row2Rank,
       unsigned int nTree);

  static unique_ptr<class TrainCtg> Classification(
       const class FrameTrain *frameTrain,
       const class RankedSet *rankedPair,
       const unsigned int *yCtg,
       const double *yProxy,
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
  unique_ptr<class LeafTrainCtg> leafCtg;
 public:

  /**
  */
  TrainCtg(const class FrameTrain *frameTrain,
	   const unsigned int *yCtg,
	   const double *yProxy,
	   unsigned int nCtg,
	   unsigned int _nTree);

  ~TrainCtg() {}
  
  class LeafTrainCtg *SubLeaf() const {
    return leafCtg.get();
  }

  class LeafTrain *Leaf() const;
};


class TrainReg : public Train {
  unique_ptr<class LeafTrainReg> leafReg;

 public:
 /**
  */
 TrainReg(const class FrameTrain *frameTrain,
	  const double *y,
	  const unsigned int *row2Rank,
	  unsigned int _nTree);

 ~TrainReg() {}
 
  class LeafTrainReg *SubLeaf() const {
    return leafReg.get();
  }

  class LeafTrain *Leaf() const;
};


#endif
