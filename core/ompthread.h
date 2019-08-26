// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file ompthread.h

   @brief Definitions for parameterization of OMP threads.

   @author Mark Seligman
 */


#ifndef CORE_OMPTHREAD_H
#define CORE_OMPTHREAD_H

#include <memory>
using namespace std;

// Some versions of OpenMP will not parallelize on unsigned types.
typedef size_t OMPBound;

/**
   @brief Static members parametrize implementation of thread parallelism.
 */
struct OmpThread {
  static unsigned int nThread;
  /**
     @brief Sets number of threads to safe value.
   */
  static void init(unsigned int nThread_);

  /**
     @brief Restores static initialization values.
   */
  static void deInit();

private:
  static constexpr unsigned int nThreadDefault = 0; // Static initialization.
  static const unsigned int maxThreads;
};

#endif
