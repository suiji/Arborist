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
   @file signatureRf.h

   @brief C++ class definitions for managing flat data frames.

   @author Mark Seligman

 */


#ifndef FRAMEMAPR_FRAME_H
#define FRAMEMAPR_FRAME_H


#include <Rcpp.h>
using namespace Rcpp;


/**
  @brief Maps factor encodings of current observation set to those of training.

  Employs proxy values for any levels unseen during training.

  @param sXFac is a (posibly empty) zero-based integer matrix.

  @param sPredMap associates (zero-based) core and front-end predictors.

  @param sLevels contain the level strings of core-indexed factor predictors.

  @param sSigTrain holds the training signature.

  @return rewritten factor matrix.
*/
RcppExport SEXP FrameReconcile(SEXP sXFac,
                               SEXP sPredMap,
                               SEXP sLevels,
                               SEXP sSigTrain);


/**
  @brief Wraps frame components supplied by front end.

  @param sX is the raw data frame, with columns either factor or numeric.

  @param sXNum is a (possibly empty numeric matrix.

  @param sXFac is a (posibly empty) zero-based integer matrix.

  @param sPredMap associates (zero-based) core and front-end predictors.

  @param sFacCard is the cardinality of core-indexed factor predictors.

  @param sLevels contain the level strings of core-indexed factor predictors.

  @return wrapped frame containing separately-typed matrices.
*/
RcppExport SEXP WrapFrame(SEXP sX,
                          SEXP sXNum,
                          SEXP sXFac,
                          SEXP sPredMap,
                          SEXP sFacCard,
                          SEXP sLevels);

RcppExport SEXP FrameNum(SEXP sX);

RcppExport SEXP FrameSparse(SEXP sX);

#endif
