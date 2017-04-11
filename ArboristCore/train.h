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
  static constexpr double slopFactor = 1.2; // Estimates tree growth.
  static unsigned int trainBlock; // Front-end defined buffer size.
  const unsigned int nTree;

  class ForestTrain *forest;
  std::vector<double> &predInfo; // E.g., Gini gain:  nPred.
  class Response *response;

  static void DeImmutables();

  /**
  */
  Train(const std::vector<unsigned int> &_yCtg, unsigned int _ctgWidth, const std::vector<double> &_yProxy, const class PMTrain *pmTrain, std::vector<unsigned int> &_origin, std::vector<unsigned int> &_facOrigin, std::vector<double> &_predInfo, std::vector<class ForestNode> &_forestNode, std::vector<unsigned int> &_facSplit, std::vector<unsigned int> &_leafOrigin, std::vector<class LeafNode> &_leafNode, std::vector<class BagLeaf> &_bagLeaf, std::vector<unsigned int> &_bagBits, std::vector<double> &_weight);

 /**
  */
  Train(const std::vector<double> &_y, const std::vector<unsigned int> &_row2Rank, const class PMTrain *pmTrain, std::vector<unsigned int> &_origin, std::vector<unsigned int> &_facOrigin, std::vector<double> &_predInfo, std::vector<class ForestNode> &_forestNode, std::vector<unsigned int> &_facSplit, std::vector<unsigned int> &_leafOrigin, std::vector<class LeafNode> &_leafNode, std::vector<class BagLeaf> &_bagLeaf, std::vector<unsigned int> &_bagBits);

  ~Train();
  
  void TrainForest(const class PMTrain *pmTrain, const class RowRank *rowRank);

 public:
/**
   @brief Static initializer.

   @return void.
 */
  static void Init(unsigned int _nPred, unsigned int _nTree, unsigned int _nSamp, const std::vector<double> &_feSampleWeight, bool withRepl, unsigned int _trainBlock, unsigned int _minNode, double _minRatio, unsigned int _totLevels, unsigned int _ctgWidth, unsigned int _predFixed, const double _splitQuant[], const double _predProb[], bool thinLeaves, const double _regMono[] = 0);

  static void Regression(const unsigned int _feRow[], const unsigned int _feRank[], const unsigned int _feNumOff[], const double _feNumVal[], const unsigned int _feRLE[], unsigned int _rleLength, const std::vector<double> &_y, const std::vector<unsigned int> &_row2Rank, std::vector<unsigned int> &_origin, std::vector<unsigned int> &_facOrigin, std::vector<double> &_predInfo, const std::vector<unsigned int> &_feCard, std::vector<class ForestNode> &_forestNode, std::vector<unsigned int> &_facSplit, std::vector<unsigned int> &_leafOrigin, std::vector<class LeafNode> &_leafNode, std::vector<class BagLeaf> &_bagLeaf, std::vector<unsigned int> &_bagBits);

  static void Classification(const unsigned int _feRow[], const unsigned int _feRank[], const unsigned int _feNumOff[], const double _feNumVal[], const unsigned int _feRLE[], unsigned int _rleLength, const std::vector<unsigned int>  &_yCtg, unsigned int _ctgWidth, const std::vector<double> &_yProxy, std::vector<unsigned int> &_origin, std::vector<unsigned int> &_facOrigin, std::vector<double> &_predInfo, const std::vector<unsigned int> &_feCard, std::vector<class ForestNode> &_forestNode, std::vector<unsigned int> &_facSplit, std::vector<unsigned int> &_leafOrigin, std::vector<class LeafNode> &_leafNode, std::vector<class BagLeaf> &_bagLeaf, std::vector<unsigned int> &_bagBits, std::vector<double> &_weight);

  void Reserve(class PreTree **ptBlock, unsigned int tCount);
  unsigned int BlockPeek(class PreTree **ptBlock, unsigned int tCount, unsigned int &blockFac, unsigned int &blockBag, unsigned int &blockLeaf, unsigned int &maxHeight);
  void BlockTree(class PreTree **ptBlock, unsigned int tStart, unsigned int tCount);
  void Block(const class RowRank *rowRank, unsigned int tStart, unsigned int tCount);
};


#endif
