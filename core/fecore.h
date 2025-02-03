// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file fecore.h

   @brief Core handshake with front-end bridge.

   @author Mark Seligman
 */

#ifndef CORE_FECORE_H
#define CORE_FECORE_H

/**
   @brief Interface class for front end.

   Maintains core-specific parameters.
*/
struct FECore {

  /**
     @brief Static initialization of core parameters.
   */
  static void init(unsigned int nThread);


  /**
     @brief Sets parallel thread count.

     @param prospective thread count.
   */
  static void setNThread(unsigned int nThread);

  
  /**
     @return available thread count.
   */
  static unsigned int getNThread();

  
  /**
     @brief Static resetting of core parameters.
   */
  static void deInit();

};

#endif
