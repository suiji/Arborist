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
#include <algorithm>

#include "typeparam.h"

class Predict {
  const unsigned int noLeaf; // Inattainable leaf index value.
 protected:
  const class FramePredict *framePredict;
  const unsigned int nTree;
  const unsigned int nRow;
  unsigned int *predictLeaves;

 public:  
  
  Predict(const class FramePredict *_framePredict,
	  unsigned int _nTree,
	  unsigned int _nRow,
	  unsigned int _noLeaf);
  virtual ~Predict();

  static void Regression(const class FramePredict *framePredict,
			 const class Forest *forest,
			 const class LeafReg *_leafReg,
			 vector<double> &_yPred);


  static void Quantiles(const class FramePredict *framePredict,
			const class Forest *forest,
			const class LeafReg *leafReg,
			vector<double> &_yPred,
			const vector<double> &quantVec,
			unsigned int qBin,
			vector<double> &qPred,
			bool validate);

  static void Classification(const class FramePredict *framePredict,
			     const class Forest *forest,
			     const class LeafCtg *leafCtg,
			     vector<unsigned int> &_yPred,
			     unsigned int *_census,
			     const vector<unsigned int> &_yTest,
			     unsigned int *_conf,
			     vector<double> &_error,
			     double *_prob);

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
  inline void LeafIdx(unsigned int blockRow,
		      unsigned int tc,
		      unsigned int leafIdx) {
    predictLeaves[nTree * blockRow + tc] = leafIdx;
  }

  
  /**
     @brief Accessor for prediction at specified coordinates.
   */
  inline unsigned int LeafIdx(unsigned int blockRow,
			      unsigned int tc) const {
    return predictLeaves[nTree * blockRow + tc];
  }


  inline const class FramePredict *PredMap() const {
    return framePredict;
  }


  inline unsigned int NoLeaf() const {
    return noLeaf;
  }
};


class PredictReg : public Predict {
  const class LeafReg *leafReg;
  vector<double> &yPred;
  double defaultScore;
  void Score(unsigned int rowStart, unsigned int rowEnd);
  
 public:
  PredictReg(const FramePredict *_framePredict,
	     const class LeafReg *_leafReg,
	     unsigned int _nTree,
	     vector<double> &_yPred);
  ~PredictReg() {}

  void PredictAcross(const class Forest *forest);
  void PredictAcross(const Forest *forest, class Quant *quant, double qPred[], bool validate);
};


class PredictCtg : public Predict {
  const class LeafCtg *leafCtg;
  const unsigned int ctgWidth;
  vector<unsigned int> &yPred;
  unsigned int defaultScore;
  vector<double> defaultWeight;
  void Validate(const vector<unsigned int> &yTest,
		unsigned int confusion[],
		vector<double> &error);
  void Vote(double *votes, unsigned int census[]);
  void Prob(double *prob, unsigned int rowStart, unsigned int rowEnd);
  void Score(double *votes, unsigned int rowStart, unsigned int rowEnd);
  unsigned int DefaultScore();
  void DefaultInit();
  double DefaultWeight(double *weightPredict);
 public:
  PredictCtg(const class FramePredict *_framePredict, const class LeafCtg *_leafCtg, unsigned int _nTree, vector<unsigned int> &_yPred);
  ~PredictCtg();

  void PredictAcross(const class Forest *forest, unsigned int *census, const vector<unsigned int> &yTest, unsigned int *conf, vector<double> &error, double *prob);
};
#endif
