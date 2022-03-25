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
   @file signature.h

   @brief C++ class definitions for managing flat data frames.

   @author Mark Seligman

 */


#ifndef DEFRAMER_SIGNATURE_H
#define DEFRAMER_SIGNATURE_H


#include <Rcpp.h>
using namespace Rcpp;

struct Signature {

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
     @brief Ensures the passed object has Frame type.

     @param frame is the object to be checked.
   */
  static SEXP checkFrame(const List& frame);


  /**
     @brief Ensures passed object contains member of class Signature.

     @param sParent is the parent object.

     @return signature object. 
   */
  static SEXP checkSignature(const List& sParent);

  
  /**
     @brief Unwraps field values useful for export.

     @param[out] level outputs all training factor levels.

     @param[out] factor outputs only realized factor levels.
   */
  static void unwrapExport(const List& sTrain,
                           List& level,
			   List& factor,
			   StringVector& names);

  /**
     @brief Unwraps level field.

     @param[out] level outputs the training factor levels, regardless of realization.

     @return List of level CharacterVectors for each categorical predictor. 
   */
  static List unwrapLevel(const List& sTrain);


  /**
     @brief Unwraps factor field.

     @param[out] level outputs the realized training factor encodings.

     @return List of realized factors for each categorical predictor. 
   */
  static List unwrapFactor(const List& sTrain);


  /**
     @brief Provides a signature for a factor-valued matrix.

     @param nPred is the number of predictors (columns).

     @param colNames are the column names.

     @param rowNames are the row names.
   */
  static List wrapFac(unsigned int nPred,
		      const CharacterVector& colNames,
		      const CharacterVector& rowNames);
  
  /**
     @brief Provides a signature for a numeric matrix.

     Parameters as above.
   */
  static List wrapNum(unsigned int nPred,
		      const CharacterVector& colNames,
		      const CharacterVector& rowNames);
  

  static List wrap(unsigned int nPred,
		   const CharacterVector& predForm,
		   const List& level,
		   const List& factor,
		   const CharacterVector& colNames,
		   const CharacterVector& rowNames);
};


#endif
