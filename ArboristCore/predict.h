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
 protected:
  static const int rowBlock = 10000;
  const int nTree;
  const unsigned int nRow;
  int *predictLeaves;

 public:  
  
  Predict(int _nTree, unsigned int _nRow);
  ~Predict();

  static void Regression(double *_blockNumT, int *_blockFacT, unsigned int _nPredNum, unsigned int _nPredFac, std::vector<class ForestNode> &_forestNode, std::vector<unsigned int> &_origin, std::vector<unsigned int> &_facOff, std::vector<unsigned int> &_facSplit, std::vector<double> &yPred, const std::vector<unsigned int> &_bag);

  static void Quantiles(double *_blockNumT, int *_blockFacT, unsigned int _nPredNum, unsigned int _nPredFac, std::vector<ForestNode> &_forestNode, std::vector<unsigned int> &_origin, std::vector<unsigned int> &_facOff, std::vector<unsigned int> &_facSplit, const std::vector<unsigned int> &rank, const std::vector<unsigned int> &sCount, const std::vector<double> &yRanked, std::vector<double> &yPred, const std::vector<double> &quantVec, unsigned int qBin, std::vector<double> &qPred, const std::vector<unsigned int> &_bag);

  static void Classification(double *_blockNumT, int *_blockFacT, unsigned int _nPredNum, unsigned int _nPredFac, std::vector<class ForestNode> &_forestNode, std::vector<unsigned int> &_origin, std::vector<unsigned int> &_facOff, std::vector<unsigned int> &_facSplit, unsigned int _ctgWidth, double *_leafWeight, std::vector<int> &yPred, int *_census, int *_yTest, int *_conf, double *_error, double *_prob, const std::vector<unsigned int> &_bag);
};


class PredictReg : public Predict {
  void Score(const class Forest *_forest, double yPred[], unsigned int rowStart, unsigned int rowEnd);
 public:
  PredictReg(int _nTree, unsigned int _nRow);
  void PredictAcross(const class Forest *forest, std::vector<double> &yPred, const class BitMatrix *bag);
  void PredictAcross(const Forest *forest, std::vector<double> &yPred, class Quant *quant, double qPred[], const BitMatrix *bag);
};


class PredictCtg : public Predict {
  const unsigned int ctgWidth;
  double *leafWeight;
  void Validate(const int yCtg[], const int yPred[], int confusion[], double error[]);
  void Vote(double *votes, int census[], int yPred[]);
  void Prob(const class Forest *_forest, double *prob, unsigned int rowStart, unsigned int rowEnd);
  void Score(const class Forest *_forest, double *votes, unsigned int rowStart, unsigned int rowEnd);
 public:
  PredictCtg(int _nTree, unsigned _nRow, unsigned int _ctgWidth, double *_leafWeight);
  void PredictAcross(Forest *forest, class BitMatrix *bag, int *census, std::vector<int> &yPred, int *yTest, int *conf, double *error, double *prob);
};
#endif
