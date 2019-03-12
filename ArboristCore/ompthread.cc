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
  #define omp_get_num_threads() 1
  #define omp_get_max_threads() 1
  #define omp_get_thread_limit() 1
  #define omp_get_thread_num() 0
  #define omp_set_nested(a) // Dummied out.
#endif

unsigned int OmpThread::nThread = OmpThread::nThreadDefault;

void OmpThread::init(unsigned int nThread_) {
  unsigned int ompMax = std::min(omp_get_max_threads(), omp_get_thread_limit());

  // Guards agains unreasonable value from system calls:
  unsigned int maxLocal = std::min(ompMax, maxThreads);
  nThread = nThread_ > 0 ? std::min(nThread_, maxLocal) : maxLocal;
}


void OmpThread::deInit() {
  nThread = nThreadDefault;
}
