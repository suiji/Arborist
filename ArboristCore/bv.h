// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file bv.h

   @brief Methods for manipulating hand-rolled bit vector.

   @author Mark Seligman

 */

#ifndef ARBORIST_BV_H
#define ARBORIST_BV_H

class BV {

  static const unsigned int slotBits = 8 * sizeof(unsigned int);
 public:

  /**
     @brief Accessor for slotwise bit count.

     @return count of bits per slot.
  */
  static unsigned int SlotBits() {
    return slotBits;
  }
  
  // Compiler should be generating shifts.  In c++11 can replace
  // with constant expression for log2(slotBits) and introduce
  // shifts explicitly.

  /**
     @brief Aligns length to the nearest containing bit slot.

     @param len is the length to align.

     @return length of containing quantity.
   */
  static unsigned int LengthAlign(unsigned int len) {
    return (len + slotBits - 1) / slotBits;
  }

  
  /**
     @brief Tests the bit at a specified position.

     @param bv is the bit vector implementation.

     @param pos is the bit position to test.

     @return true iff bit position is set in the bit vector.
   */
  static inline bool IsSet(const unsigned int bv[], unsigned int pos) {
    unsigned int slot = pos / slotBits;
    unsigned int mask = 1 << (pos - (slot * slotBits));

    return (bv[slot] & mask) != 0;
  }


  /**
     @brief Sets the bit at position 'pos'.

     @param bv is the bit vector implementation.

     @param pos is the position to set.

     @return void.
   */
  static inline void SetBit (unsigned int bv[], unsigned int pos) {
    unsigned int slot = pos / slotBits;
    unsigned int mask = 1 << (pos - (slot * slotBits));
    bv[slot] |= mask;
  }
  
};


#endif
