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
  const unsigned int noLeaf; // Inattainable leaf index value.
 protected:
  class PMPredict *pmPredict;
  const unsigned int nTree;
  const unsigned int nRow;
  unsigned int *predictLeaves;

 public:  
  
  Predict(class PMPredict *_pmPredict, unsigned int _nTree, unsigned int _nRow, unsigned int _noLeaf);
  virtual ~Predict();

  static void Regression(const std::vector<double> &valNum, const std::vector<unsigned int> &rowStart, const std::vector<unsigned int> &runLength, const std::vector<unsigned int> &_predStart, double *_blockNumT, unsigned int *_blockFacT, unsigned int _nPredNum, unsigned int _nPredFac, std::vector<class ForestNode> &_forestNode, std::vector<unsigned int> &_origin, std::vector<unsigned int> &_facOff, std::vector<unsigned int> &_facSplit, std::vector<unsigned int> &_leafOrigin, std::vector<class LeafNode> &_leafNode, std::vector<class BagLeaf> &_bagLeaf, std::vector<unsigned int> &_bagBits, const std::vector<double> &yTrain, std::vector<double> &_yPred);


  static void Quantiles(const std::vector<double> &valNum, const std::vector<unsigned int> &rowStart, const std::vector<unsigned int> &runLength, const std::vector<unsigned int> &_predStart, double *_blockNumT, unsigned int *_blockFacT, unsigned int _nPredNum, unsigned int _nPredFac, std::vector<ForestNode> &_forestNode, std::vector<unsigned int> &_origin, std::vector<unsigned int> &_facOff, std::vector<unsigned int> &_facSplit, std::vector<unsigned int> &_leafOrigin, std::vector<LeafNode> &_leafNode, std::vector<class BagLeaf> &_bagLeaf, std::vector<unsigned int> &_bagBits, const std::vector<double> &yTrain, std::vector<double> &_yPred, const std::vector<double> &quantVec, unsigned int qBin, std::vector<double> &qPred, bool validate);

  static void Classification(const std::vector<double> &valNum, const std::vector<unsigned int> &rowStart, const std::vector<unsigned int> &runLength, const std::vector<unsigned int> &_predStart, double *_blockNumT, unsigned int *_blockFacT, unsigned int _nPredNum, unsigned int _nPredFac, std::vector<class ForestNode> &_forestNode, std::vector<unsigned int> &_origin, std::vector<unsigned int> &_facOff, std::vector<unsigned int> &_facSplit, std::vector<unsigned int> &_leafOrigin, std::vector<LeafNode> &_leafNode, std::vector<class BagLeaf> &_bagLeaf, std::vector<unsigned int> &_bagBits, std::vector<double> &_leafInfoCtg, std::vector<unsigned int> &_yPred, unsigned int *_census, const std::vector<unsigned int> &_yTest, unsigned int *_conf, std::vector<double> &_error, double *_prob);

  const double *RowNum(unsigned int row) const;
  const unsigned int *RowFac(unsigned int row) const;
  

  /**
     @brief Assigns a proxy leaf index at the prediction coordinates passed.

     @return void.
   */
  inline void BagIdx(unsigned int blockRow, unsigned int tc) {
    predictLeaves[nTree * blockRow + tc] = noLeaf;
  }

  
  /**
   */
  inline bool IsBagged(unsigned int blockRow, unsigned int tc) const {
    return predictLeaves[nTree * blockRow + tc] == noLeaf;
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

  inline const class PMPredict *PredMap() const {
    return pmPredict;
  }
};


class PredictReg : public Predict {
  const class LeafReg *leafReg;
  const std::vector<double> &yTrain;
  std::vector<double> &yPred;
  double defaultScore;
  void Score(unsigned int rowStart, unsigned int rowEnd);
  double DefaultScore();
 public:
  PredictReg(PMPredict *_pmPredict, const class LeafReg *_leafReg, const std::vector<double> &_yTrain, unsigned int _nTree, std::vector<double> &_yPred);
  ~PredictReg() {}

  void PredictAcross(const class Forest *forest);
  void PredictAcross(const Forest *forest, class Quant *quant, double qPred[], bool validate);

  
  /**
     @brief Access to training response vector.

     @return constant reference to training response vector.
   */
  inline const std::vector<double> &YTrain() const {
    return yTrain;
  }
};


class PredictCtg : public Predict {
  const class LeafCtg *leafCtg;
  const unsigned int ctgWidth;
  std::vector<unsigned int> &yPred;
  unsigned int defaultScore;
  double *defaultWeight;
  void Validate(const std::vector<unsigned int> &yTest, unsigned int confusion[], std::vector<double> &error);
  void Vote(double *votes, unsigned int census[]);
  void Prob(double *prob, unsigned int rowStart, unsigned int rowEnd);
  void Score(double *votes, unsigned int rowStart, unsigned int rowEnd);
  unsigned int DefaultScore();
  double DefaultWeight(double *weightPredict);
 public:
  PredictCtg(class PMPredict *_pmPredict, const class LeafCtg *_leafCtg, unsigned int _nTree, std::vector<unsigned int> &_yPred);
  ~PredictCtg();

  void PredictAcross(const class Forest *forest, unsigned int *census, const std::vector<unsigned int> &yTest, unsigned int *conf, std::vector<double> &error, double *prob);
};
#endif
