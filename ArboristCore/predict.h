// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file predict.h

   @brief Data structures and methods for prediction.

   @author Mark Seligman
 */

#ifndef ARBORIST_PREDICT_H
#define ARBORIST_PREDICT_H

#include <vector>

class Predict {
  const unsigned int nonLeafIdx; // Inattainable leaf index value.
 protected:
  static const int rowBlock = 8192;
  const int nTree;
  const unsigned int nRow;
  unsigned int *predictLeaves;

 public:  
  
  Predict(int _nTree, unsigned int _nRow, unsigned int _nonLeafIdx);
  ~Predict();

  static void Regression(double *_blockNumT, int *_blockFacT, unsigned int _nPredNum, unsigned int _nPredFac, std::vector<class ForestNode> &_forestNode, std::vector<unsigned int> &_origin, std::vector<unsigned int> &_facOff, std::vector<unsigned int> &_facSplit, std::vector<unsigned int> &_leafOrigin, std::vector<class LeafNode> &_leafNode, std::vector<class BagRow> &_bagRow, std::vector<unsigned int> &_rank, std::vector<double> &yPred, unsigned int bagTrain);


  static void Quantiles(double *_blockNumT, int *_blockFacT, unsigned int _nPredNum, unsigned int _nPredFac, std::vector<ForestNode> &_forestNode, std::vector<unsigned int> &_origin, std::vector<unsigned int> &_facOff, std::vector<unsigned int> &_facSplit, std::vector<unsigned int> &_leafOrigin, std::vector<LeafNode> &_leafNode, std::vector<class BagRow> &_bagRow, std::vector<unsigned int> &_rank, const std::vector<double> &yRanked, std::vector<double> &yPred, const std::vector<double> &quantVec, unsigned int qBin, std::vector<double> &qPred, unsigned int bagTrain);

  static void Classification(double *_blockNumT, int *_blockFacT, unsigned int _nPredNum, unsigned int _nPredFac, std::vector<class ForestNode> &_forestNode, std::vector<unsigned int> &_origin, std::vector<unsigned int> &_facOff, std::vector<unsigned int> &_facSplit, std::vector<unsigned int> &_leafOrigin, std::vector<LeafNode> &_leafNode, std::vector<class BagRow> &_bagRow, std::vector<double> &_leafInfoCtg, std::vector<int> &yPred, int *_census, const std::vector<unsigned int> &_yTest, int *_conf, std::vector<double> &_error, double *_prob, unsigned int bagTrain);

  /**
     @brief Assigns a proxy leaf index at the prediction coordinates passed.

     @return void.
   */
  inline void BagIdx(unsigned int blockRow, unsigned int tc) {
    predictLeaves[nTree * blockRow + tc] = nonLeafIdx;
  }

  
  /**
   */
  inline bool IsBagged(unsigned int blockRow, unsigned int tc) const {
    return predictLeaves[nTree * blockRow + tc] == nonLeafIdx;
  }


  /**
     @brief Assigns a true leaf index at the prediction coordinates passed.

     @return void.
   */
  inline void LeafIdx(unsigned int blockRow, unsigned int tc, unsigned int leafIdx) {
    predictLeaves[nTree * blockRow + tc] = leafIdx;
  }

  
  /**
     @brief Accessor for prediction at specified coordinates.
   */
  inline unsigned int LeafIdx(unsigned int blockRow, unsigned int tc) const {
    return predictLeaves[nTree * blockRow + tc];
  }
};


class PredictReg : public Predict {
  const class LeafReg *leafReg;
  void Score(unsigned int rowStart, unsigned int rowEnd, double yPred[]);
 public:
  PredictReg(const class LeafReg *_leafReg, int _nTree, unsigned int _nRow, unsigned int _nonLeafIdx);
  void PredictAcross(const class Forest *forest, std::vector<double> &yPred, const class BitMatrix *bag);
  void PredictAcross(const Forest *forest, std::vector<double> &yPred, class Quant *quant, double qPred[], const BitMatrix *bag);
};


class PredictCtg : public Predict {
  const class LeafCtg *leafCtg;
  const unsigned int ctgWidth;
  void Validate(const std::vector<unsigned int> &yTest, const int yPred[], int confusion[], std::vector<double> &error);
  void Vote(double *votes, int census[], int yPred[]);
  void Prob(double *prob, unsigned int rowStart, unsigned int rowEnd);
  void Score(double *votes, unsigned int rowStart, unsigned int rowEnd);
 public:
  PredictCtg(const class LeafCtg *_leafCtg, int _nTree, unsigned _nRow, unsigned int _nonLeafIdx);
  void PredictAcross(const class Forest *forest, const class BitMatrix *bag, int *census, std::vector<int> &yPred, const std::vector<unsigned int> &yTest, int *conf, std::vector<double> &error, double *prob);
};
#endif
