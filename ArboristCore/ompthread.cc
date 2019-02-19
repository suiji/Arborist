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

#include "omp.h"
#include <algorithm>

unsigned int OmpThread::threadStart = OmpThread::nThreadDefault;

void OmpThread::init(unsigned int nThread_) {
  threadStart = omp_get_max_threads();
  unsigned int nThread;
  if (nThread_ > 0) {
    nThread = nThread_;
  }
  else { // Guards agains unreasonable value from system call.
    nThread = std::min(threadStart, maxThreads);
  }
  omp_set_num_threads(nThread);
}


void OmpThread::deInit() {
  omp_set_num_threads(threadStart);
  threadStart = nThreadDefault;
}
