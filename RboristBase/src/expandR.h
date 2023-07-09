// Copyright (C)  2012-2023  Mark Seligman
//
// This file is part of RboristBase.
//
// RboristBase is free software: you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 2 of the License, or
// (at your option) any later version.
//
// RboristBase is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with RboristBase.  If not, see <http://www.gnu.org/licenses/>.


/**
   @file expandR.h

   @brief Expands trained forest into a collection of vectors.

   @author Mark Seligman
 */


#ifndef EXPAND_R_H
#define EXPAND_R_H

#include <vector>
#include <memory>
using namespace std;

#include <Rcpp.h>
using namespace Rcpp;


/**
   @brief Expands trained forest into summary vectors.

   @param sTrain is the trained forest.

   @return expanded forest as list of vectors.
 */
RcppExport SEXP expandR(SEXP sTrain);


struct ExpandR {

  /**
     @brief Wraps expanded values for classification leaves.

     @param tIdx is the tree index.

     @return wrapped leaf scores.
  */
  static List expandLeafCtg(const struct LeafExpandCtg& leaf,
                            unsigned int tIdx);


  /**
     @brief As above, but regression leaves.
   */
  static List expandLeafReg(const struct LeafExpandReg& leaf,
                            unsigned int tIdx);


  static List expandForest(const class ForestExpand& forestExpand,
                           unsigned int tIdx);


  static IntegerVector expandBag(const struct SamplerExpand& sampler,
				 const struct LeafExpand& leaf,
                                 unsigned int tIdx);


  static List expandTreeReg(const List& sTrain,
                            const IntegerVector& predMap);


  static List expandTreeCtg(const List& lTrain,
			    const IntegerVector& predMap);


  static List expandReg(const List& sTrain);


  static List expandCtg(const List& sTrain);
};


#endif
