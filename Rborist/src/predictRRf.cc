// Copyright (C)  2012-2025  Mark Seligman
//
// This file is part of Rborist.
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
   @file predictRRf.cc

   @brief C++ interface to R prediction entry for Rborist package.

   @author Mark Seligman
 */

#include "predictR.h"
#include "predictbridge.h"
#include "corebridge.h"


// [[Rcpp::export]]
void PredictR::initPerInvocation(const List& lArgs) {
  PredictBridge::initPredict(as<bool>(lArgs[strIndexing]),
			     as<bool>(lArgs[strBagging]),
			     as<unsigned int>(lArgs[strImpPermute]),
			     as<bool>(lArgs[strTrapUnobserved]));
  PredictBridge::initQuant(quantVec(lArgs));
  PredictBridge::initCtgProb(as<bool>(lArgs[strCtgProb]));
  CoreBridge::init(as<unsigned int>(lArgs[strNThread]));
}
