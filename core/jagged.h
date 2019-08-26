// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file jagged.h

   @brief Templated definitions for jagged-array containers:  irregular
   major stride provided by vector lookup, with unit minor stride.

   @author Mark Seligman
 */

#ifndef CORE_JAGGED_H
#define CORE_JAGGED_H

#include <vector>
using namespace std;

template<class item_type, class off_type>
class Jagged {
protected:
  const unsigned int nMajor; // Major dimension.

public:
  off_type height; // Major offsets.
  item_type items;

  
  /**
     @brief Returns the item count.
   */
  virtual size_t size() const = 0;


  /**
     @brief Returns the base offset associated with the major dimension.
   */
  virtual size_t majorOffset(unsigned int maj) const = 0;

  Jagged(const unsigned int nMajor_,
         off_type height_,
         item_type items_) :
    nMajor(nMajor_),
    height(height_),
    items(items_) {
  }

  virtual ~Jagged() {}


  unsigned int getNMajor() const {
    return nMajor;
  }

  auto getHeight(unsigned int idx) const {
    return height[idx];
  }
};


/**
   @brief Two-dimensional jagged array.
   Height records row high-watermarks.
 */
template <class item_type, class off_type>
class JaggedArrayBase : public Jagged<item_type, off_type> {
public:

  size_t majorOffset(unsigned int maj) const {
    return maj == 0 ? 0 : Jagged<item_type, off_type>::height[maj-1];
  }


  size_t absOffset(unsigned int maj, unsigned int idx) const {
    return majorOffset(maj) + idx;
  }

  JaggedArrayBase(const unsigned int nMajor_,
                  off_type height_,
                  item_type items_) :
    Jagged<item_type, off_type>(nMajor_, height_, items_) {
  }

  ~JaggedArrayBase() {}

  size_t size() const {
    return Jagged<item_type, off_type>::height[Jagged<item_type, off_type>::nMajor - 1];
  }
};


/**
   @brief Unspecialized variant inheriting from base.
 */
template<class item_type, class off_type>
class JaggedArray : public JaggedArrayBase<item_type, off_type> {
public:
  JaggedArray(const unsigned int nMajor_,
              off_type height_,
              item_type items_) :
    JaggedArrayBase<item_type, off_type>(nMajor_, height_, items_) {
  }

  ~JaggedArray() {}
};


template <class item_type, class off_type>
class Jagged3Base : public JaggedArrayBase<item_type, off_type> {
protected:
  const unsigned int stride;

public:
  Jagged3Base(const unsigned int stride_,
              const unsigned int nMajor_,
              off_type height_,
              item_type items_) :
    JaggedArrayBase<item_type, off_type>(nMajor_, height_, items_), stride(stride_) {
  }

  ~Jagged3Base() {}

  /**
     @brief Returns minor base offset associated with leaf coordinate.
   */
  size_t minorOffset(unsigned int maj,
                     unsigned int dim2) const {
    return JaggedArrayBase<item_type, off_type>::absOffset(maj, stride * dim2);
  }
};


template<class item_type, class off_type>
class Jagged3 : public Jagged3Base<item_type, off_type> {
public:
  Jagged3(const unsigned int stride_,
          const unsigned int nMajor_,
          off_type height_,
          item_type items_) :
    Jagged3Base<item_type, off_type>(stride_, nMajor_, height_, items_) {
  }

  ~Jagged3() {}
};

#endif
