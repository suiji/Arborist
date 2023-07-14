// Copyright (C)  2012-2023   Mark Seligman
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
   @file leafR.cc

   @brief C++ interface to terminal map.

   @author Mark Seligman
 */


#include "resizeR.h"
#include "leafbridge.h"
#include "samplerR.h"
#include "samplerbridge.h"
#include "leafR.h"

const string LeafR::strExtent = "extent";
const string LeafR::strIndex = "index";


LeafR::LeafR() :
  extent(NumericVector(0)),
  index(NumericVector(0)),
  extentTop(0),
  indexTop(0) {
}


void LeafR::bridgeConsume(const LeafBridge& bridge,
			  double scale) {

  size_t extentSize = bridge.getExtentSize();
  if (extentTop + extentSize > static_cast<size_t>(extent.length())) {
    extent = std::move(ResizeR::resize<NumericVector>(extent, extentTop, extentSize, scale));
  }
  bridge.dumpExtent(&extent[extentTop]);
  extentTop += extentSize;

  size_t indexSize = bridge.getIndexSize();
  if (indexTop + indexSize > static_cast<size_t>(index.length())) {
    index = std::move(ResizeR::resize<NumericVector>(index, indexTop, indexSize, scale));
  }
  bridge.dumpIndex(&index[indexTop]);
  indexTop += indexSize;
}


List LeafR::wrap() {
  BEGIN_RCPP

  List leaf = List::create(_[strExtent] = std::move(extent),
			   _[strIndex] = std::move(index)
			);
  leaf.attr("class") = "Leaf";

  return leaf;
  END_RCPP
}


LeafBridge LeafR::unwrap(const List& lTrain,
				     const SamplerBridge& samplerBridge) {
  List lLeaf((SEXP) lTrain["leaf"]);
  bool empty = (Rf_isNull(lLeaf[strIndex]) || Rf_isNull(lLeaf[strExtent]));
  bool thin = empty || as<NumericVector>(lLeaf[strExtent]).length() == 0;
  return LeafBridge(samplerBridge, thin,
		    empty ? nullptr : as<NumericVector>(lLeaf[strExtent]).begin(),
		    empty ? nullptr : as<NumericVector>(lLeaf[strIndex]).begin());
}
