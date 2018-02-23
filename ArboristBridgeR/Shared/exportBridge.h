// Copyright (C)  2012-2018  Mark Seligman
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
   @file exportBridge.h

   @brief C++ class definitions for managing class export serializtion.

   @author Mark Seligman

 */


#ifndef ARBORIST_EXPORT_BRIDGE_H
#define ARBORIST_EXPORT_BRIDGE_H

#include <Rcpp.h>
using namespace Rcpp;

RcppExport SEXP ForestFloorExport(SEXP sArbOut);

namespace ExportBridge {
  static List ExportReg(const SEXP sForest,
			const List &leaf,
			IntegerVector &predMap,
			unsigned int &nTree);


  static List ExportCtg(const SEXP sForest,
			const List &leaf,
			IntegerVector &predMap,
			unsigned int &nTree);


  static SEXP FFloorLeafReg(List &forestCore,
			    unsigned int tIdx);

  static SEXP FFloorLeafCtg(List &forestCore,
			    unsigned int tIdx);

  static SEXP FFloorInternal(List &forestCore,
			     unsigned int tIdx);

  static SEXP FFloorBag(List &forestCore,
			int tIdx);

  static List FFloorTreeReg(SEXP sForest,
			    const List &leaf,
			    IntegerVector &predMap);

  static List FFloorTreeCtg(List &coreCtg,
			    unsigned int tIdx);

  static SEXP FFloorReg(SEXP sForest,
			List &leaf,
			IntegerVector &predMap,
			List &predLevel);

  static SEXP FFloorCtg(SEXP sForest,
			List &leaf,
			IntegerVector &predMap,
			List &predLevel);
};

using namespace ExportBridge;

#endif
