// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file coproc.h

   @brief Definitions for the classes managing coprocessor support.

   @author Mark Seligman
 */

#ifndef CORE_COPROC_H
#define CORE_COPROC_H

#include "typeparam.h"

#include <string>
#include <vector>

class Coproc {
  const unsigned int nCoproc; // Number of coprocessors detected.
  const unsigned int unroll; // Unroll factor.

 public:
  static unique_ptr<Coproc> Factory(bool enable, vector<string> &diag);

 Coproc(unsigned int _nCoproc, unsigned int _unroll) : nCoproc(_nCoproc), unroll(_unroll) {
  }

  inline unsigned int getUnroll() const {
    return unroll;
  }
  
  inline unsigned int getNCoproc() const {
    return nCoproc;
  }
  
};

#endif
