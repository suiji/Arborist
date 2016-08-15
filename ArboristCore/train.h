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
  static unsigned int nTree;
  static unsigned int nRow;
  static unsigned int nPred;

  class Forest *forest;
  double *predInfo; // E.g., Gini gain:  nPred.
  class Response *response;

  static void DeImmutables();

  /**
  */
  Train(const std::vector<unsigned int> &_yCtg, unsigned int _ctgWidth, const std::vector<double> &_yProxy, std::vector<unsigned int> &_origin, std::vector<unsigned int> &_facOrigin, double _predInfo[], std::vector<class ForestNode> &_forestNode, std::vector<unsigned int> &_facSplit, std::vector<unsigned int> &_leafOrigin, std::vector<class LeafNode> &_leafNode, std::vector<class BagRow> &_bagRow, std::vector<double> &_weight);

 /**
  */
  Train(const std::vector<double> &_y, const std::vector<unsigned int> &_row2Rank, std::vector<unsigned int> &_origin, std::vector<unsigned int> &_facOrigin, double _predInfo[], std::vector<class ForestNode> &_forestNode, std::vector<unsigned int> &_facSplit, std::vector<unsigned int> &_leafOrigin, std::vector<class LeafNode> &_leafNode, std::vector<class BagRow> &_bagRow, std::vector<unsigned int> &_rank);

  ~Train();
  
  void ForestTrain(const class RowRank *rowRank);

 public:
/**
   @brief Static initializer.

   @return void.
 */
  static void Init(const double _feNum[], const unsigned int _facCard[], unsigned int _cardMax, unsigned int _nPredNum, unsigned int _nPredFac, unsigned int _nRow, unsigned int _nTree, unsigned int _nSamp, const double _feSampleWeight[], bool withRepl, unsigned int _trainBlock, unsigned int _minNode, double _minRatio, unsigned int _totLevels, unsigned int _ctgWidth, unsigned int _predFixed, const double _predProb[], const double _regMono[] = 0);

  static void Regression(unsigned int _feRow[], unsigned int _feRank[], unsigned int _feInvNum[], const std::vector<double> &_y, const std::vector<unsigned int> &_row2Rank, std::vector<unsigned int> &_origin, std::vector<unsigned int> &_facOrigin, double _predInfo[], std::vector<class ForestNode> &_forestNode, std::vector<unsigned int> &_facSplit, std::vector<unsigned int> &_leafOrigin, std::vector<class LeafNode> &_leafNode, std::vector<class BagRow> &_bagRow, std::vector<unsigned int> &_rank);

  static void Classification(unsigned int _feRow[], unsigned int _feRank[], unsigned int _feInvNum[], const std::vector<unsigned int>  &_yCtg, unsigned int _ctgWidth, const std::vector<double> &_yProxy, std::vector<unsigned int> &_origin, std::vector<unsigned int> &_facOrigin, double _predInfo[], std::vector<class ForestNode> &_forestNode, std::vector<unsigned int> &_facSplit, std::vector<unsigned int> &_leafOrigin, std::vector<class LeafNode> &_leafNode, std::vector<class BagRow> &_bagRow, std::vector<double> &_weight);

  void Reserve(class PreTree **ptBlock, unsigned int tCount);
  unsigned int BlockPeek(class PreTree **ptBlock, unsigned int tCount, unsigned int &blockFac, unsigned int &blockBag, unsigned int &blockLeaf, unsigned int &maxHeight);
  void BlockTree(class PreTree **ptBlock, unsigned int tStart, unsigned int tCount);
  void Block(const class RowRank *rowRank, unsigned int tStart, unsigned int tCount);
};


#endif
