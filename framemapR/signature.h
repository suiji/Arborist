// Copyright (C)  2012-2019  Mark Seligman
//
// This file is part of framemapR.
//
// framemapR is free software: you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 2 of the License, or
// (at your option) any later version.
//
// framemapR is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with framemapR.  If not, see <http://www.gnu.org/licenses/>.

/**
   @file signature.h

   @brief C++ class definitions for managing flat data frames.

   @author Mark Seligman

 */


#ifndef FRAMEMAPR_SIGNATURE_H
#define FRAMEMAPR_SIGNATURE_H


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

     @param[out] predMap outputs the core predictor mapping.

     @param[out] level outputs the training factor levels.
   */
  static void unwrapExport(const List& sTrain,
                           IntegerVector& predMap,
                           List& level);

  /**
     @brief Unwraps level field.

     @param[out] level outputs the training factor levels.

     @return List of level CharacterVectors for each categorical predictor. 
   */
  static List unwrapLevel(const List& sTrain);



  static SEXP wrapSignature(const IntegerVector& predMap,
                 const List& level,
                 const CharacterVector& colNames,
                 const CharacterVector& rowNames);

};


#endif
