// Copyright (C)  2012-2025  Mark Seligman
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


#include <Rcpp.h>
using namespace Rcpp;

/**
   @file rcppInit.h

   @brief C++ interface to R entry for symbol registration.

   @author Mark Seligman
 */


/**
   @brief Lights off symbol registry for package loading.

   @param info is an external handle for symbol registration.

   @return void.
 */
void R_init_Rborist(DllInfo *info);
