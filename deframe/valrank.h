// This file is part of deframe.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file valrank.h

   @brief Type-templated classes for ranking vectors.

   @author Mark Seligman
 */


#ifndef DEFRAME_VALRANK_H
#define DEFRAME_VALRANK_H

#include "typeparam.h" // For now

#include <algorithm>
#include <vector>
using namespace std;

/**
   @brief value/row pair workspace for ranking.
 */
template<typename valType>
struct ValRow {
  valType val;
  size_t row;
  IndexT rank; // For now.

  void init(valType val,
	    size_t row) {
    this->val = val;
    this->row = row;
    rank = 0; // Assigned separately.
  }

  void setRank(const ValRow& predec) {
    rank = (val == predec.val) ? predec.rank : 1 + predec.rank;
  }
};


template<typename valType>
class ValRank {
  const size_t nRow;
  vector<ValRow<valType> > valRow;

public:

  ValRank(const valType val[],
          size_t nRow_) : nRow(nRow_),
                          valRow(vector<ValRow<valType> >(nRow)) {
    size_t row = 0;
    for (auto & vr : valRow) {
      vr.init(val[row], row);
      row++;
    }
    order();
  }


  /**
     @brief Accessor for row count.
   */
  auto getNRow() const {
    return nRow;
  }


  auto getRow(size_t idx) const {
    return valRow[idx].row;
  }
  
  /**
     @brief Accessor for value at a given row.

     @return looked up value.
   */
  valType getVal(size_t idx) const {
    return valRow[idx].val;
  }


  auto getRank(size_t idx) const {
    return valRow[idx].rank;
  }


  /**
     @return number of distinct rank values.
   */
  auto getRankCount() const {
    return valRow[valRow.size() - 1].rank + 1;
  }
  
  
  /**
     @brief Orders and assigns ranks.
     
     Ensures a stable sort ut identify maximal runs.

     N.B.:  extraneous parentheses work around parser error in older g++.
   */
  void order() {
    sort(valRow.begin(), valRow.end(), [] (const ValRow<valType>& a,
                                           const ValRow<valType>& b) -> bool {
                                         return (a.val < b.val) || ((a.val == b.val) && ((a.row) < b.row));
                                       }
      );

    // Increments rank values beginning from default value of zero at base.
    //
    for (size_t idx = 1; idx < nRow; idx++) {
      valRow[idx].setRank(valRow[idx-1]);
    }
  }


  /**
     @brief Presents ranks in row order.

     @return vector mapping row indices to ranks.
   */
  vector<IndexT> rank() const {
    vector<IndexT> row2Rank(nRow);
    for (auto vr : valRow) {
      row2Rank[vr.row] = vr.rank;
    }
    return row2Rank;
  }
};

#endif
