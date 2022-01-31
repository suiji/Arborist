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


#ifndef RF_RESIZE_R_H
#define RF_RESIZE_R_H

#include <Rcpp.h>
using namespace Rcpp;

/**
   @file resizeR.h

   @brief Static vector resizing methods which should be templated.

   @author Mark Seligman
 */


struct ResizeR {

  static RawVector resizeRaw(const RawVector& raw,
			     size_t offset,
			     size_t count,
			     double scale) { // Assumes scale >= 1.0.
    RawVector temp(scale * (offset + count));
    for (size_t i = 0; i < offset; i++)
      temp[i] = raw[i];

    return temp;
  }


  static NumericVector resizeNum(const NumericVector& num,
				 size_t offset,
				 size_t count,
				 double scale) { // Assumes scale >= 1.0.
    NumericVector temp(scale * (offset + count));
    for (size_t i = 0; i < offset; i++)
      temp[i] = num[i];

    return temp;
  }


  static ComplexVector resizeComplex(const ComplexVector& num,
				     size_t offset,
				     size_t count,
				     double scale) { // Assumes scale >= 1.0.
    ComplexVector temp(scale * (offset + count));
    for (size_t i = 0; i < offset; i++)
      temp[i] = num[i];

    return temp;
  }
};

#endif
