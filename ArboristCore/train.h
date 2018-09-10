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

  const unsigned int nRow;
  const unsigned int treeChunk;
  unique_ptr<class BitMatrix> bagRow;
  unique_ptr<class ForestTrain> forest;
  vector<double> predInfo; // E.g., Gini gain:  nPred.
  const unique_ptr<class Response> response;

  void TrainForest(const class FrameTrain *frameTrain,
                   const class RankedSet *rankedPair);

 public:

  virtual void Leaves(class Sample *sample, const vector<unsigned int> &leafMap, unsigned int blockIdx) const = 0;
  virtual void Reserve(unsigned int leafEst, unsigned int bagEst) const = 0;

  /**
     @brief Regression constructor.
  */
  Train(const class FrameTrain *frameTrain,
        const double *y,
        const unsigned int *row2Rank,
        unsigned int treeChunk_);

  
  /**
     @brief Classification constructor.
  */
  Train(const class FrameTrain *frameTrain,
        const unsigned int *yCtg,
        const double *yProxy,
        unsigned int treeChunk_);

  virtual ~Train();


  const vector<double> &getPredInfo() const {
    return predInfo;
  }

  const unsigned int NRow() const {
    return nRow;
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
  
  static void InitSample(unsigned int nSamp);

  static void InitCtgWidth(unsigned int ctgWidth);

  static void InitSplit(unsigned int minNode,
                        unsigned int totLevels,
                        double minRatio);
  
  static void InitMono(const class FrameTrain* frameTrain,
                       const vector<double> &regMono);

  static void DeInit();

  static unique_ptr<class TrainReg> regression(
       const class FrameTrain *frameTrain,
       const class RankedSet *rankedPair,
       const double *y,
       const unsigned int *row2Rank,
       unsigned int treeChunk);


  static unique_ptr<class TrainCtg> classification(
       const class FrameTrain *frameTrain,
       const class RankedSet *rankedPair,
       const unsigned int *yCtg,
       const double *yProxy,
       unsigned int nCtg,
       unsigned int treeChunk,
       unsigned int nTree);
  
  void Reserve(vector<TrainPair> &treeBlock);
  unsigned int blockPeek(vector<TrainPair> &treeBlock,
                         unsigned int &blockFac,
                         unsigned int &blockBag,
                         unsigned int &blockLeaf,
                         unsigned int &maxHeight);
  void blockConsume(vector<TrainPair> &treeBlock,
                    unsigned int blockStart);
  void treeBlock(const class FrameTrain *frameTrain,
                 const class RowRank *rowRank,
                 unsigned int tStart,
                 unsigned int tCount);

  class ForestTrain *getForest() const {
    return forest.get();
  }

  /**
     @param[out] bbRaw
   */
  void getBag(unsigned char bbRaw[]) const;
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
           unsigned int treeChunk,
           unsigned int nTree);

  ~TrainCtg();

  class LeafTrainCtg *getLeaf() const {
    return leafCtg.get();
  }

  void Leaves(class Sample *sample, const vector<unsigned int> &leafMap, unsigned int blockIdx) const;

  void Reserve(unsigned int leafEst, unsigned int bagEst) const;
};


class TrainReg : public Train {
  unique_ptr<class LeafTrainReg> leafReg;

 public:
 /**
  */
 TrainReg(const class FrameTrain *frameTrain,
          const double *y,
          const unsigned int *row2Rank,
          unsigned int treeChunk_);

 ~TrainReg();

 class LeafTrainReg *getLeaf() const {
   return leafReg.get();
 };

 void Leaves(class Sample *sample, const vector<unsigned int> &leafMap, unsigned int blockIdx) const;
 void Reserve(unsigned int leafEst, unsigned int bagEst) const;
};


#endif
