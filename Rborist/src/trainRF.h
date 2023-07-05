// Copyright (C)  2012-2023   Mark Seligman
//
// This file is part of Rborist
//
// Rborist is free software: you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 2 of the License, or
// (at your option) any later version.
//
// Rborist is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with rfR.  If not, see <http://www.gnu.org/licenses/>.

/**
   @file trainRf.h

   @brief C++ interface to R entry for RF training.

   @author Mark Seligman
 */

#ifndef RBORIST_TRAINRF_H
#define RBORIST_TRAINRF_H

#include <Rcpp.h>
using namespace Rcpp;


/**
   @brief Main training entry from front end.
 */
RcppExport SEXP trainRF(const SEXP sRLEFrame,
			const SEXP sSampler,
			const SEXP sArgList);

#endif
