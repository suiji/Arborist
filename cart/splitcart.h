// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#ifndef CART_SPLITCART_H
#define CART_SPLITCART_H


/**
   @file splitcart.h

   @brief Static entry points for CART-specific node splitting.

   @author Mark Seligman
 */


struct SplitCart {
  static unique_ptr<class SplitFrontier> factory(class Frontier* frontier);
};


#endif

