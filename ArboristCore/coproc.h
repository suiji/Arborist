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

#ifndef ARBORIST_COPROC_H
#define ARBORIST_COPROC_H

#include <string>
#include <vector>

class Coproc {
  const unsigned int nCoproc; // Number of coprocessors detected.
  const unsigned int unroll; // Unroll factor.

 public:
  static Coproc* Factory(bool enable, std::string &diag);

 Coproc(unsigned int _nCoproc, unsigned int _unroll) : nCoproc(_nCoproc), unroll(_unroll) {
  }

  inline unsigned int NCoproc() const {
    return nCoproc;
  }
  
};

#endif
