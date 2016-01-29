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

#include <vector>
//using namespace std;

/**
   @brief Interface class for front end.  Holds simulation-specific parameters
   of the data and constructs forest, leaf and diagnostic structures.
*/
class Train {
 protected:
  static int trainBlock; // Front-end defined buffer size.
  static int nTree;
  static unsigned int nRow;
  static int nPred;

  std::vector<int> &inBag; // Packed vector:  treated as unsigned[].
  int *orig; // Tree origins:  nTree.
  int *facOrig; // Factor bit origins:  nTree.
  double *predInfo; // E.g., Gini gain:  nPred.
  std::vector<int> &pred;
  std::vector<double> &split;
  std::vector<int> &bump;
  std::vector<unsigned int> &facSplit;

  static void DeImmutables();

  Train(std::vector<int> &_inBag, int _orig[], int _facOrig[], double _predInfo[], std::vector<int> &_pred, std::vector<double> &_split, std::vector<int> &_bump, std::vector<unsigned int> &_facSplit);

  virtual ~Train() {};
  void Forest(const class RowRank *rowRank);
  void BagSetTree(const unsigned int bagSource[], int treeNum);


 public:
/**
   @brief Static initializer.

   @return void.
 */
  static void Init(double *_feNum, int _facCard[], int _cardMax, int _nPredNum, int _nPredFac, int _nRow, int _nTree, int _nSamp, double _feSampleWeight[], bool withRepl, int _trainBlock, int _minNode, double _minRatio, int _totLevels, int _ctgWidth, int _predFixed, double _predProb[], int _regMono[] = 0);

  static void Regression(int _feRow[], int _feRank[], int _feInvNum[], double _y[], double _yRanked[], std::vector<int> &_inBag, int _origin[], int _facOrig[], double _predInfo[], std::vector<int> &_pred, std::vector<double> &_split, std::vector<int> &_bump, std::vector<unsigned int> &_facSplit, std::vector<unsigned int> &_rank, std::vector<unsigned int> &_sCount);

  static void Classification(int _feRow[], int _feRank[], int _feInvNum[], int _yCtg[], int _ctgWidth, double _yProxy[], std::vector<int> &_inBag, int _orig[], int _facOrig[], double _predInfo[], std::vector<int> &_pred, std::vector<double> &_split, std::vector<int> &_bump, std::vector<unsigned int> &_facSplit, std::vector<double> &_weight);

  void Reserve(class PreTree **ptBlock, int tCoun);
  int BlockPeek(class PreTree **ptBlock, int tCount, int &blockFac, int &blockBag, int &maxHeight);
  void BlockTree(class PreTree **ptBlock, int tStart, int tCount);
  void Grow(unsigned int height, unsigned int bitWidth);
  virtual void LeafReserve(int heightEst, int bagEst) = 0;
  virtual void Block(const class RowRank *rowRank, int tStart, int tCount) = 0;
};


class TrainReg : public Train {
  std::vector<unsigned int> &rank;
  std::vector<unsigned int> &sCount;
  class ResponseReg *responseReg;
  void Block(const class RowRank *rowRank, int tStart, int tCount);
  void LeafReserve(int heightEst, int bagEst);
  void BlockLeaf(class PreTree **ptBlock, class SampleReg **sampleBlock, int tSTart, int tCount, int tOrig);
 public:
  TrainReg(double _y[], double _yRanked[], std::vector<int> &_inBag, int _orig[], int _facOrig[], double _predInfo[], std::vector<int> &_pred, std::vector<double> &_split, std::vector<int> &_bump, std::vector<unsigned int> &_facSplit, std::vector<unsigned int> &_rank, std::vector<unsigned int> &_sCount);
  ~TrainReg();
};


class TrainCtg : public Train {
  const unsigned int ctgWidth;
  std::vector<double> &weight;
  class ResponseCtg *responseCtg;
  void Block(const class RowRank *rowRank, int tStart, int tCount);
  void LeafReserve(int heightEst, int bagEst);
  void BlockLeaf(class PreTree **ptBlock, class SampleCtg **sampleBlock, int tStart, int tCount, int tOrig);
 public:
  TrainCtg(int _yCtg[], unsigned int _ctgWidth, double _yProxy[], std::vector<int> &_inBag, int _orig[], int _facOrig[], double _predInfo[], std::vector<int> &_pred, std::vector<double> &_split, std::vector<int> &_bump, std::vector<unsigned int> &_facSplit, std::vector<double> &_weight);
  ~TrainCtg();
};

#endif
