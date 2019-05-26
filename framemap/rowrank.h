// This file is part of framemap.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file rowrank.h

   @brief Class pairing row index and rank/category at that row.

   @author Mark Seligman
 */

#ifndef FRAMEMAP_ROWRANK_H
#define FRAMEMAP_ROWRANK_H

class RowRank {
 protected:
  unsigned int row;
  unsigned int rank;

 public:
  RowRank() {}
  
  RowRank(unsigned int row_,
          unsigned int rank_) : row(row_),
                                rank(rank_) {
  }
  
  void init(unsigned int row, unsigned int rank) {
    this->row = row;
    this->rank = rank;
  }

  inline auto getRow() const {
    return row;
  }


  inline auto getRank() const {
    return rank;
  }
};

#endif

