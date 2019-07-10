// Copyright (C)  2012-2019   Mark Seligman
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
   @file callback.h

   @brief Exposes utility functions implemented by the front end.

   @author Mark Seligman
 */

#ifndef ARBORIST_CALLBACK_H
#define ARBORIST_CALLBACK_H

#include <vector>
using namespace std;

struct CallBack {
  /**
    @brief Call-back to Rcpp implementation of row sampling.

    @param nSamp is the number of samples to draw.

    @return copy of sampled row indices.
  */
  static vector<unsigned int> sampleRows(unsigned int nSamp);


  /**
    @brief Call-back to R's uniform random-variate generator.

    @param len is number of variates to generate.

    @return std::vector copy of R-generated random variates.
  */
  static vector<double> rUnif(size_t len);
};

#endif
