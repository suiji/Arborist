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


class Predict {
 protected:
  const unsigned int nRow;
  const unsigned int nPredNum;
  const unsigned int nPredFac;
  const int nTree;
  double *blockNumT;
  int *blockFacT;
  int *predictLeaves;

  void PredictAcross(class Forest *forest, const unsigned int *bag);
 public:  
  
  Predict(unsigned int _nRow, int _nTree, unsigned int _nPredNum, unsigned int _nPredFac, double *_blockNumT, int *_blockFacT);
  ~Predict();

  static void Regression(double *_blockNumT, int *_blockFacT, unsigned int _nRow, unsigned int _nPredNum, unsigned int _nPredFac, int _nTree, int _forestSize, int _preds[], double _splits[], int _bump[], int _origins[], int _facOff[], unsigned int _facSplit[], double yPred[], const unsigned int *bag);

  static void Quantiles(double *_blockNumT, int *_blockFacT, unsigned int _nRow, unsigned int _nPredNum, unsigned int _nPredFac, int _nTree, int _forestSize, int _preds[], double _splits[], int _bump[], int _origins[], int _facOff[], unsigned int _facSplit[], unsigned int _rank[], unsigned int _sCount[], double _yRanked[], double _yPred[], double _quantVec[], int _qCount, unsigned int _qBin, double _qPred[], const unsigned int *_bag);

  static void Classification(double *_blockNumT, int *_blockFacT, unsigned int _nRow, unsigned int _nPredNum, unsigned int _nPredFac, int _nTree, int _forestSize, int _preds[], double _splits[], int _bump[], int _origins[], int _facOff[], unsigned int _facSplit[], unsigned int _ctgWidth, double *_leafWeight, int *_yPred, int *_census, int *_yTest, int *_conf, double *_error, double *_prob, unsigned int *_bag);

  inline unsigned int NRow() {
    return nRow;
  }


  inline int NPredNum() {
    return nPredNum;
  }


  inline int NPredFac() {
    return nPredFac;
  }

  /**
     @return base address for numeric values at row.
   */
  inline double *RowNum(unsigned int row) {
    return &blockNumT[nPredNum * row];
  }


  /**
     @return base address for factor values at row.
   */
  inline int *RowFac(unsigned int row) {
    return &blockFacT[nPredFac * row];
  }
};


class PredictReg : public Predict {
 public:
  PredictReg(unsigned int _nRow, int _nTree, unsigned int _nPredNum, unsigned int _nPredFac, double *_blockNumT, int *_blockFacT);
  void Score(double yPred[], class Forest *_forest);
};


class PredictCtg : public Predict {
  const unsigned int ctgWidth;
  double *leafWeight;
 public:
  PredictCtg(unsigned int _nRow, int _nTree, unsigned int _nPredNum, unsigned int _nPredFac, double *_blockNumT, int *_blockFacT, unsigned int _ctgWidth, double *_leafWeight);
  void Validate(const int yCtg[], const int yPred[], int confusion[], double error[]);
  void Vote(double *votes, int census[], int yPred[]);
  void Prob(double *prob, class Forest *_forest);
  double *Score(class Forest *_forest);
};
#endif
