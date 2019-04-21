// Copyright (C)  2012-2019  Mark Seligman
//
// This file is part of rfR.
//
// rfR is free software: you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 2 of the License, or
// (at your option) any later version.
//
// rfR is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with rfR.  If not, see <http://www.gnu.org/licenses/>.

/**
   @file exportRf.h

   @brief C++ class definitions for managing class export serializtion.

   @author Mark Seligman

 */


#ifndef ARBORIST_EXPORT_RF_H
#define ARBORIST_EXPORT_RF_H

#include <Rcpp.h>
using namespace Rcpp;

RcppExport SEXP ForestFloorExport(SEXP sArbOut);

struct ExportRf {

  static List fFloorLeafReg(const class LeafRegRf *leaf,
                            unsigned int tIdx);

  static List fFloorLeafCtg(const class LeafCtgRf *leaf,
                            unsigned int tIdx);

  static List fFloorForest(const class ForestExport *forestExport,
                           unsigned int tIdx);

  static IntegerVector fFloorBag(const class LeafRf *leaf,
                                 unsigned int tIdx,
                                 unsigned int rowTrain);

  static List fFloorTreeReg(const List &sTrain,
                            IntegerVector &predMap);

  static List fFloorTreeCtg(const class ForestExport *forest,
                            const class LeafCtgRf *leaf,
                            unsigned int rowTrain);

  static List fFloorReg(const List& sTrain,
                        IntegerVector &predMap,
                        List &predLevel);

  static List fFloorCtg(const List& sTrain,
                        IntegerVector &predMap,
                        List &predLevel);
};

#endif
