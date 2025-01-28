// Copyright (C)  2012-2024  Mark Seligman
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
   @file signatureR.h

   @brief C++ class definitions for managing flat data frames.

   @author Mark Seligman

 */


#ifndef DEFRAMER_SIGNATURE_R_H
#define DEFRAMER_SIGNATURE_R_H


#include <Rcpp.h>
using namespace Rcpp;

using namespace std;


/**
   @brief reconciles column ordering in new data.

   @param sDF references a data frame.

   @param sSigTrain is the signature of the training frame.

   @param sKeyed indicates whether keyed access is requested.

   @return permuted column order if keyable, otherwise 1::length.
 */
RcppExport SEXP columnOrder(const SEXP sDF,
			    const SEXP sSigTrain,
			    const SEXP sKeyed);


/**
   @brief R-language encapsulation of a frame signature.

   Signatures contain front-end annotations not exposed to core.
   Column and row names stubbed to zero-length vectors if null.
 */
struct SignatureR {
  static const string strClassName; ///< Deframer name for signature.
  static const string strColName; ///< Predictor names.  May be null.
  static const string strRowName; ///< Observation names.  Often null.
  static const string strPredLevel; ///< Per-predictor levels.
  static const string strPredFactor; ///< Per-predictor realized levels.
  static const string strPredType; ///< Per-predictor type name.
  static const string strFactorType; ///< What R calls factor types.
  static const string strNumericType; ///< What R calls numeric types.
  
  /**
     @brief Derives or creates vector of row names for frame.

     @param sFrame contains the parent Frame.

     @return vector of row names.
   */
  static CharacterVector unwrapRowNames(const List& sFrame);


  /**
     @return vector of column (predictor) names.
   */
  static CharacterVector unwrapColNames(const List& sFrame);


  /**
     @return contents of named field, if nonempty.
   */
  static CharacterVector unwrapName(const List& signature,
				    const string& name);


  /**

     @brief Checks whether new frame's types coform to training frame.

     @return false iff training signature nonempty and not matching.
  */
  static bool checkTypes(SEXP sSigTrain,
			 const CharacterVector& predClass);


  /**
     @brief Ensures the passed object has Deframe type.

     @param deFrame is the object to be checked.

     @return true iff deFrame has type Deframe.
   */
  static bool checkFrame(const List& deFrame);


  /**
     @brief Checks whether signature supports keyed access.

     @return true iff column names unique and non-null.
   */
  static bool checkKeyable(const List& sigTrain);


  /**
     @brief Ensures deframed object contains Signature.

     @param lParent is an object containing signature.

     @return signature object. 
   */
  static List getSignature(const List& lParent);


  /**
     @brief Unwraps level field.

     @param[out] level outputs the training factor levels, regardless of realization.

     @return List of level CharacterVectors for each categorical predictor. 
   */
  static List unwrapLevel(const List& sTrain);


  /**
     @brief Unwraps factor field.

     @param[out] level outputs the realized training factor encodings.

     @return List of realized levels for each categorical predictor. 
   */
  static List getFactor(const List& lTrain);


  /**
     @brief As above, but gets all levels.
   */
  static List getLevel(const List& lTrain);


  /**
     @brief lParent is an object containing a Signature member.
     
     @return count of factor-valued predictors.
   */
  static unsigned int nFactor(const List& lParent);


  /**
     @return mapping from core index to original position.
   */
  static IntegerVector predMap(const List& lTrain);

  
  /**
     @return number of predictors involved in training.
   */
  static unsigned int nPred(const List& lTrain);


  /**
     @brief Provides a signature for a factor-valued matrix.
   */
  static List wrapFactor(const IntegerMatrix& blockFac);


  /**
     @brief Provides a signature for a numeric matrix.
   */
  static List wrapNumeric(const NumericMatrix& blockNum);


  /**
     @brief Provides a signature for a sparse matrix.

     @param isFactor is true iff the matrix values are categorical.
   */
  static List wrapSparse(unsigned int nPred,
			 bool isFactor,
			 const CharacterVector& colNames,
			 const CharacterVector& rowNames);

  /**
     @brief Provides a signature for a mixed data frame.
   */
  static List wrapMixed(unsigned int nPred,
			const CharacterVector& predClass,
			const List& level,
			const List& factor,
			const CharacterVector& colNames,
			const CharacterVector& rowNames);


  static List wrapDF(const DataFrame& df,
		     const CharacterVector& predClass,
		     const List& lLevel,
		     const List& lFactor);
};

#endif
