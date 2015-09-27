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
  static unsigned int ctgWidth;

  std::vector<int> &inBag; // Packed vector:  treated as unsigned[].
  int *orig; // Tree origins:  nTree.
  int *facOrig; // Factor bit origins:  nTree.
  double *predInfo; // E.g., Gini gain:  nPred.
  std::vector<int> &pred;
  std::vector<double> &split;
  std::vector<int> &bump;
  std::vector<int> &facSplit;

  static void Immutables(int _nTree, int _nRow, int _nPred, int _nSamp, int _trainBlock, int _minNode, double _minRatio, int _totLevels, unsigned int _ctgWidth);
  static void DeImmutables();

  Train(std::vector<int> &_inBag, int _orig[], int _facOrig[], double _predInfo[], std::vector<int> &_pred, std::vector<double> &_split, std::vector<int> &_bump, std::vector<int> &_facSplit);

  virtual ~Train() {};
  void Forest(const class PredOrd *predOrd);
  void BagSetTree(const unsigned int bagSource[], int treeNum);


 public:
  static void Init(int _nTree, int _nRow, int _nPred, int _nSamp, int trainBlock, int _minNode, double _minRatio, int _totLevels, unsigned int _ctgWidth = 0);
  static void ForestReg(double _y[], double _yRanked[], std::vector<int> &_inBag, int _origin[], int _facOrig[], double _predInfo[], std::vector<int> &_pred, std::vector<double> &_split, std::vector<int> &_bump, std::vector<int> &_facSplit, std::vector<unsigned int> &_rank, std::vector<unsigned int> &_sCount);
  static void ForestCtg(int _yCtg[], double _yProxy[], std::vector<int> &_inBag, int _orig[], int _facOrig[], double _predInfo[], std::vector<int> &_pred, std::vector<double> &_split, std::vector<int> &_bump, std::vector<int> &_facSplit, std::vector<double> &_weight);

  void Reserve(class PreTree **ptBlock, int tCoun);
  int BlockPeek(class PreTree **ptBlock, int tCount, int &blockFac, int &blockBag, int &maxHeight);
  void BlockTree(class PreTree **ptBlock, int tStart, int tCount);
  void Grow(unsigned int height, unsigned int bitWidth);
  virtual void LeafReserve(int heightEst, int bagEst) = 0;
  virtual void Block(const class PredOrd *predOrd, int tStart, int tCount) = 0;
};


class TrainReg : public Train {
  std::vector<unsigned int> &rank;
  std::vector<unsigned int> &sCount;
  class ResponseReg *responseReg;
  void Block(const class PredOrd *predOrd, int tStart, int tCount);
  void LeafReserve(int heightEst, int bagEst);
  void BlockLeaf(class PreTree **ptBlock, class SampleReg **sampleBlock, int tSTart, int tCount, int tOrig);
 public:
  TrainReg(double _y[], double _yRanked[], std::vector<int> &_inBag, int _orig[], int _facOrig[], double _predInfo[], std::vector<int> &_pred, std::vector<double> &_split, std::vector<int> &_bump, std::vector<int> &_facSplit, std::vector<unsigned int> &_rank, std::vector<unsigned int> &_sCount);
  ~TrainReg();
};


class TrainCtg : public Train {
  std::vector<double> &weight;
  class ResponseCtg *responseCtg;
  void Block(const class PredOrd *predOrd, int tStart, int tCount);
  void LeafReserve(int heightEst, int bagEst);
  void BlockLeaf(class PreTree **ptBlock, class SampleCtg **sampleBlock, int tStart, int tCount, int tOrig);
 public:
  TrainCtg(int _yCtg[], double _yProxy[], std::vector<int> &_inBag, int _orig[], int _facOrig[], double _predInfo[], std::vector<int> &_pred, std::vector<double> &_split, std::vector<int> &_bump, std::vector<int> &_facSplit, std::vector<double> &_weight);
  ~TrainCtg();
};

#endif
