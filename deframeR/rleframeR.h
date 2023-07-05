// Copyright (C)  2012-2022  Mark Seligman
//
// This file is part of deframeR.
//
// deframeR is free software: you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 2 of the License, or
// (at your option) any later version.
//
// deframeR is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with deframeR.  If not, see <http://www.gnu.org/licenses/>.

/**
   @file rleframeR.h

   @brief C++ class definitions for managing RankedFrame object.

   @author Mark Seligman

 */


#ifndef DEFRAMER_RLEFRAMER_H
#define DEFRAMER_RLEFRAMER_H

#include <Rcpp.h>
using namespace Rcpp;

#include <vector>
#include <memory>
using namespace std;

#include "rleframe.h"
#include "block.h"


/**
   @brief Methods for caching and consuming RLE frame representation.
 */
struct RLEFrameR {
  /**
     @brief Checks that front end provides valid RankedFrame representation.

     @return List object containing valid representation.
   */
  static List checkRankedFrame(SEXP sRankedFrame);

  /**
     @brief Checks that front end provides valid NumRanked representation.

     @return List object containing valid representation.
   */
  static List checkNumRanked(SEXP sNumRanked);

  /**
     @brief As above, but checks factor representation.
   */
  static List checkFacRanked(SEXP sFacRanked);


  /**
     @brief  Checks whether a frame supports keyed access.

     @return true iff training column names match uniquely.
   */
  static bool checkKeyable(const DataFrame& df,
			   const List& sigTrain);


  /**
     @brief Sorts data frame in blocks of like type.

     @param df is the data frame.

     @param lSigTrain is a training signature, possibly null.

     @param lLevel are factor levels, if any.

     @param predClass are the type name strings, per predictor.
   */
  static List presortDF(const DataFrame& df,
		        SEXP sSigTrain,
			SEXP sLevel,
			const CharacterVector& predClass);


  /**
     @brief Maps factor encodings of current observation set to those of training.

     Employs proxy values for any levels unseen during training.

     @param df is a data frame.

     @param sLevel contain the level strings of core-indexed factor predictors.

     @param sSigTrain holds the training signature.

     @return rewritten data frame.
  */
  static IntegerMatrix factorReconcile(const DataFrame& df,
				       const List& lSigTrain,
				       const List& lLevel);


  static IntegerVector columnReconcile(const IntegerVector& dfCol,
				       const CharacterVector& colTest,
				       const CharacterVector& colTrain);


  /**
     @brief Presorts a block of numeric values.

     @param sX is the front end's matrix representation.

     @return R-style list of sorting summaries.
   */
  static List presortNum(SEXP sX);


  /**
     @brief Presorts a block of factor values.

     Parameters as above.
   */
  static List presortFac(SEXP sX);


  /**
     @brief Presorts a dcgMatrix encoded with 'I' and 'P' descriptors.
   */
  static List presortIP(const class BlockIPCresc<double>* rleCrescIP,
			size_t nRow,
			unsigned int nPred);

  /**
     @brief Produces an R-style run-length encoding of the frame.

     @param rleCresc is the crescent encoding.
   */
  static List wrap(const class RLECresc* rleCresc);

  
  static List wrapRF(const class RLECresc* rleCresc);


  static List wrapNum(const class RLECresc* rleCresc);

  
  static List wrapFac(const class RLECresc* rleCresc);

  
  static unique_ptr<RLEFrame> unwrap(const List& lDeframe);


  static unique_ptr<RLEFrame> unwrapFrame(const List& rankedFrame,
					  const NumericVector& numVal,
					  const IntegerVector& numHeight,
					  const IntegerVector& facVal,
					  const IntegerVector& facHeight);
};

#endif
