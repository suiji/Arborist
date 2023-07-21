// Copyright (C)  2012-2022   Mark Seligman
// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file prng.h

   @brief Interface to front end's pseudo-random variate generation.

   @author Mark Seligman
 */

#ifndef CORE_PRNG_H
#define CORE_PRNG_H

#include <vector>
using namespace std;

namespace PRNG {
  /**
    @brief Call-back to front-end session's uniform PRNG.

    @param len is number of variates to generate.

    @param scale is a coefficient by which the variates are scaled.

    @return std::vector copy of front end-generated random variates.
  */
  vector<double> rUnif(size_t len,
		       double scale = 1.0);


  /**
     @brief Derives and scales uniform index variates.

     @param scale specifies a size by which to multiply.

     @return scaled copy of random variates, as index vector.
   */
  vector<size_t> rUnifIndex(size_t len,
			    size_t scale);


  /**
     @brief Derives and scales uniform variates.

     @param scale specifies the values by which to multiply.

     @return scaled copy of random variates, as index vector.
   */
  vector<size_t> rUnifIndex(const vector<size_t>& scale);
}

#endif
