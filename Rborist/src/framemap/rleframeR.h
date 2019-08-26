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
   @file rleframeR.h

   @brief C++ class definitions for managing RankedFrame object.

   @author Mark Seligman

 */


#ifndef FRAMEMAPR_RLEFRAME_R_H
#define FRAMEMAPR_RLEFRAME_R_H

#include "rleframe.h"
#include "summaryframe.h"

#include <Rcpp.h>
using namespace Rcpp;

#include <memory>
using namespace std;

/**
   @brief External entry to presorting RankedFrame builder.

   @param sFrame is an R-style List containing frame block.

   @return R-style representation of run-length encoding.
 */
RcppExport SEXP Presort(SEXP sFrame);


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
     @brief Unwraps an R-style run-length encoding.

     @param autoCompress is the percentage threshold for predictor sparsity.

     @param coproc summarizes the coprocessor configuration, if any.

     @return summary-style representation of frame.
   */
  static unique_ptr<RLEFrame> factory(SEXP sRLEFrame,
                                      unsigned int nRow);

  static unique_ptr<RLEFrame> factory(const IntegerVector& card,
                                      size_t nRow,
                                      const RawVector& rle,
                                      const NumericVector& numVal,
                                      const IntegerVector& numOff);
  /**
     @brief Static entry to block sorting.

     @param frame summarizes the predictor blocking scheme.

     @return R-style list of sorting summaries.
   */
  static List presort(const List& frame);


  /**
     @brief Produces an R-style run-length encoding of the frame.

     @param rleCresc is the crescent encoding.
   */
  static List wrap(const class RLECresc* rleCresc);
};

#endif
