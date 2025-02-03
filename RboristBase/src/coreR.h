// Copyright (C)  2012-2025   Mark Seligman
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
   @file coreR.h

   @brief C++ interface to core parameter manipulation.

   @author Mark Seligman
 */

#ifndef CORE_R_H
#define CORE_R_H

#include <Rcpp.h>
using namespace Rcpp;

using namespace std;

RcppExport SEXP setThreadCount(SEXP sNThread);

#endif
