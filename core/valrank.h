// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file valrow.h

   @brief Type-templated classes for ranking vectors.

   @author Mark Seligman
 */


#ifndef CORE_VALRANK_H
#define CORE_VALRANK_H

#include <algorithm>
#include <vector>
using namespace std;

/**
   @brief value/row pair workspace for ranking.
 */
template<typename tn>
struct ValRow {
  tn val;
  size_t row;
  unsigned int rank; // For now.

  void init(tn val, size_t row) {
    this->val = val;
    this->row = row;
    rank = 0; // Assigned separately.
  }

  void setRank(const ValRow& predec) {
    rank = (val == predec.val) ? predec.rank : 1 + predec.rank;
  }
};


template<typename tn>
class ValRank {
  const size_t nRow;
  vector<ValRow<tn> > valRow;

public:

  ValRank(const tn val[],
          size_t nRow_) : nRow(nRow_),
                          valRow(vector<ValRow<tn> >(nRow)) {
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
  tn getVal(size_t idx) const {
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
    sort(valRow.begin(), valRow.end(), [] (const ValRow<tn>& a,
                                           const ValRow<tn>& b) -> bool {
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
  vector<unsigned int> rank() const {
    vector<unsigned int> row2Rank(nRow);
    for (auto vr : valRow) {
      row2Rank[vr.row] = vr.rank;
    }
    return row2Rank;
  }
};

#endif
