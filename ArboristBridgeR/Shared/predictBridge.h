// Copyright (C)  2012-2018   Mark Seligman
//
// This file is part of ArboristBridgeR.
//
// ArboristBridgeR is free software: you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 2 of the License, or
// (at your option) any later version.
//
// ArboristBridgeR is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with ArboristBridgeR.  If not, see <http://www.gnu.org/licenses/>.

/**
   @file predictBridge.h

   @brief C++ interface to R entry for prediction.

   @author Mark Seligman
 */


#ifndef ARBORIST_PREDICT_BRIDGE_H
#define ARBORIST_PREDICT_BRIDGE_H

#include <Rcpp.h>
using namespace Rcpp;

#include "predict.h"

RcppExport SEXP ValidateReg(SEXP sPredBlock,
				  SEXP sForest,
				  SEXP sLeaf,
				  SEXP sYTest);

RcppExport SEXP TestReg(SEXP sPredBlock,
			      SEXP sForest,
			      SEXP sLeaf,
			      SEXP sYTest);

RcppExport SEXP ValidateVotes(SEXP sPredBlock,
				    SEXP sForest,
				    SEXP sLeaf,
				    SEXP sYTest);

RcppExport SEXP ValidateProb(SEXP sPredBlock,
				   SEXP sForest,
				   SEXP sLeaf,
				   SEXP sYTest);

RcppExport SEXP ValidateQuant(SEXP sPredBlock,
				    SEXP sForest,
				    SEXP sLeaf,
				    SEXP sYTest,
				    SEXP sQuantVec,
				    SEXP sQBin);

RcppExport SEXP TestQuant(SEXP sPredBlock,
			       SEXP sForest,
			       SEXP sLeaf,
			       SEXP sQuantVec,
			       SEXP sQBin,
			       SEXP sYTest);

RcppExport SEXP TestProb(SEXP sPredBlock,
			      SEXP sForest,
			      SEXP sLeaf,
			      SEXP sYTest);


/**
   @brief Predicts with class votes.

   @param sPredBlock contains the blocked observations.

   @param sForest contains the trained forest.

   @param sLeaf contains the trained leaves.

   @param sVotes outputs the vote predictions.

   @return Prediction object.
 */
RcppExport SEXP TestVotes(SEXP sPredBlock,
				SEXP sForest,
				SEXP sLeaf,
				SEXP sYTest);


namespace PredictBridge {

  static List Reg(SEXP sPredBlock,
		  SEXP sForest,
		  SEXP sLeaf,
		  SEXP sYTest,
		  bool validate);

  static List SummaryReg(SEXP sYTest,
			 vector<double> &yPred,
			 NumericMatrix &qPred);
  
  static List Ctg(SEXP sPredBlock,
		  SEXP sForest,
		  SEXP sLeaf,
		  SEXP sYTest,
		  bool validate,
		  bool doProb);

  static List Quant(SEXP sPredBlock,
		    SEXP sForest,
		    SEXP sLeaf,
		    SEXP sQuantVec,
		    SEXP sQBin,
		    SEXP sYTest,
		    bool validate);

  static double MSE(const double yValid[],
		    NumericVector y,
		    double &rsq,
		    double &mae);
};



#endif
