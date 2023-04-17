// Copyright (C)  2012-2022  Mark Seligman
//
// This file is part of rf.
//
// rf is free software: you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 2 of the License, or
// (at your option) any later version.
//
// rf is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with rfR.  If not, see <http://www.gnu.org/licenses/>.


/**
   @file exportR.h

   @brief Expands trained forest into a collection of vectors.

   @author Mark Seligman
 */


#ifndef RF_EXPORT_R_H
#define RF_EXPORT_R_H

#include <vector>
#include <memory>
using namespace std;

#include <Rcpp.h>
using namespace Rcpp;

/**
   @brief Expands traind forest into summary vectors.

   @param sTrain is the trained forest.

   @return RboristExport as List.
 */
RcppExport SEXP expandRf(SEXP sTrain);


struct ExportRf {

  /**
     @brief Wraps exported values for classification leaves.

     @param tIdx is the tree index.

     @return wrapped leaf scores.
  */
  static List exportLeafCtg(const struct LeafExportCtg& leaf,
                            unsigned int tIdx);


  /**
     @brief As above, but regression leaves.
   */
  static List exportLeafReg(const struct LeafExportReg& leaf,
                            unsigned int tIdx);


  static List exportForest(const class ForestExport& forestExport,
                           unsigned int tIdx);


  static IntegerVector exportBag(const struct SamplerExport& sampler,
				 const struct LeafExport& leaf,
                                 unsigned int tIdx);


  static List exportTreeReg(const List& sTrain,
                            const IntegerVector& predMap);


  static List exportTreeCtg(const List& lTrain,
			    const IntegerVector& predMap);


  static List exportReg(const List& sTrain);


  static List exportCtg(const List& sTrain);
};


#endif
