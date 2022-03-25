// Copyright (C)  2012-2022   Mark Seligman
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
   @file prng.h

   @brief Exposes pseudo-random variate generation utilities.

   Allows core to call PRNG currently specified by the R session.

   @author Mark Seligman
 */

#ifndef CALLBACK_PRNG_H
#define CALLBACK_PRNG_H

#include <Rcpp.h>
using namespace Rcpp;

#include <vector>
using namespace std;

namespace PRNG {
  /**
    @brief Call-back to R session's uniform PRNG.

    @param len is number of variates to generate.

    @return std::vector copy of R-generated random variates.
  */
  vector<double> rUnif(size_t len,
		       double scale = 1.0);

  
  /**
   @brief Internal implementation of sampling.

   Essentially a reworking of Nathan Russell's 2016 implementation
   for Rcpp.

   Arguments as with caller.
 */
  NumericVector sampleUniform(size_t nObs,
			      size_t nSamp,
			      bool replace);
};

#endif
