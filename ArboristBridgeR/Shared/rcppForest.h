// Copyright (C)  2012-2016  Mark Seligman
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
   @file rcppForest.h

   @brief C++ class definitions for managing Forest object.

   @author Mark Seligman

 */


#ifndef ARBORIST_RCPP_FOREST_H
#define ARBORIST_RCPP_FOREST_H

using namespace Rcpp;

RcppExport SEXP RcppForestWrap(std::vector<int> pred, std::vector<double> split, std::vector<int> bump, IntegerVector origin, IntegerVector facOrigin, std::vector<unsigned int> facSplit);
void RcppForestUnwrap(SEXP sForest, int *&_pred, double *&_split, int *&_bump, int *&_origin, int *&_facOrig, unsigned int *&_facSplit, int &_nTree, int &_height);

#endif
