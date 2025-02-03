// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file corebridge.h

   @brief Front-end wrappers for setting general parameters.

   @author Mark Seligman
 */

#ifndef CORE_BRIDGE_COREBRIDGE_H
#define CORE_BRIDGE_COREBRIDGE_H

/**
   @brief Parameter-setting methods.
 */
struct CoreBridge {
  /**
     @brief Initializes general parameters.

     @param nThread is a maximum count of simultaneous threads.
   */
  static void init(unsigned int nThread);

  
  /**
     @brief Sets thread count.

     @param nThread is the prospective thread count to set.

     @return actual thread count set.
   */
  static unsigned int setNThread(unsigned int nThread);


  static void deInit();
};

#endif
