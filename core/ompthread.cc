// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file ompthread.cc

   @brief Parametrization of OMP control.

   @author Mark Seligman
 */

#include "ompthread.h"

#include <algorithm>

// Cribbed from data.table.
#ifdef _OPENMP
  #include <omp.h>
#else

constexpr int omp_get_max_threads() {
  return 1;
}

constexpr int omp_get_thread_limit() {
  return 1;
}
#endif

unsigned int OmpThread::nThread = OmpThread::nThreadDefault;

const unsigned int OmpThread::maxThreads = 1024; // Cribbed from above.


void OmpThread::init(unsigned int nThread_) {
  unsigned int ompMax = std::min(omp_get_max_threads(), omp_get_thread_limit());

  // Guards agains unreasonable value from system calls:
  unsigned int maxLocal = std::min(ompMax, maxThreads);
  nThread = nThread_ > 0 ? std::min(nThread_, maxLocal) : maxLocal;
}


void OmpThread::deInit() {
  nThread = nThreadDefault;
}
