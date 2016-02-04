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
  const unsigned int nRow;
  const int nTree;
  int *predictLeaves;

 public:  
  
  Predict(unsigned int _nRow, int _nTree);
  ~Predict();

  static void Regression(double *_blockNumT, int *_blockFacT, unsigned int _nRow, unsigned int _nPredNum, unsigned int _nPredFac, std::vector<unsigned int> &_pred, std::vector<double> &_split, std::vector<unsigned int> &_bump, std::vector<unsigned int> &_origin, const std::vector<unsigned int> &_facOff, const std::vector<unsigned int> &_facSplit, double _yPred[], const std::vector<unsigned int> &_bag);

  static void Quantiles(double *_blockNumT, int *_blockFacT, unsigned int _nRow, unsigned int _nPredNum, unsigned int _nPredFac, std::vector<unsigned int> &_pred, std::vector<double> &_split, std::vector<unsigned int> &_bump, std::vector<unsigned int> &_origin, const std::vector<unsigned int> &_facOff, const std::vector<unsigned int> &_facSplit, unsigned int _rank[], unsigned int _sCount[], double _yRanked[], double _yPred[], double _quantVec[], int _qCount, unsigned int _qBin, double _qPred[], const std::vector<unsigned int> &_bag);

  static void Classification(double *_blockNumT, int *_blockFacT, unsigned int _nRow, unsigned int _nPredNum, unsigned int _nPredFac, std::vector<unsigned int> &_pred, std::vector<double> &_split, std::vector<unsigned int> &_bump, std::vector<unsigned int> &_origin, const std::vector<unsigned int> &_facOff, const std::vector<unsigned int> &_facSplit, unsigned int _ctgWidth, double *_leafWeight, int *_yPred, int *_census, int *_yTest, int *_conf, double *_error, double *_prob, const std::vector<unsigned int> &_bag);
};


class PredictReg : public Predict {
 public:
  PredictReg(unsigned int _nRow, int _nTree);
  void Score(double yPred[], const class Forest *_forest);
};


class PredictCtg : public Predict {
  const unsigned int ctgWidth;
  double *leafWeight;
  void Validate(const int yCtg[], const int yPred[], int confusion[], double error[]);
  void Vote(double *votes, int census[], int yPred[]);
  void Prob(double *prob, const class Forest *_forest);
  double *Score(const class Forest *_forest);
 public:
  PredictCtg(unsigned int _nRow, int _nTree, unsigned int _ctgWidth, double *_leafWeight);
  void PredictAcross(Forest *forest, class BitMatrix *bag, int *census, int *yPred, int *yTest, int *conf, double *error, double *prob);
};
#endif
