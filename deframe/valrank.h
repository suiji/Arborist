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
struct ValRank {
  valType val;
  size_t row;
  IndexT rank; // For now.


  ValRank(valType val_,
	 size_t row_) : val(val_), row(row_), rank(0) {
  }

  
  ~ValRank() = default;

  /**
     @brief Sets current rank based on predecessor.
   */
  void setRank(const ValRank& predec) {
    rank = predec.rank + (areEqual(val, predec.val) ? 0 : 1);
  }
};


template<typename valType>
bool ValRankCompare (const ValRank<valType>& a,
		     const ValRank<valType>& b) {
  return (a.val < b.val) || ((a.val == b.val) && ((a.row) < b.row));
}


template<>
inline bool ValRankCompare (const ValRank<double>& a,
			    const ValRank<double>& b) {
  return (a.val < b.val) || (areEqual(a.val, b.val) && ((a.row) < b.row)) || (!isnan(a.val) && isnan(b.val));
}


template<typename valType>
class RankedObs {
  vector<ValRank<valType> > valRow;

public:

  RankedObs(const valType val[],
	    size_t nRow) {
    for (size_t row = 0; row < nRow; row++) {
      valRow.emplace_back(val[row], row);
    }
    order();
  }


  /**
     @brief Accessor for row count.
   */
  auto getNRow() const {
    return valRow.size();
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
    sort(valRow.begin(), valRow.end(), ValRankCompare<valType>);

    // Increments rank values beginning from default value of zero at base.
    //
    for (size_t idx = 1; idx < valRow.size(); idx++) {
      valRow[idx].setRank(valRow[idx-1]);
    }
  }


  /**
     @brief Presents ranks in row order.

     @return vector mapping row indices to ranks.
   */
  vector<IndexT> rank() const {
    vector<IndexT> row2Rank(valRow.size());
    for (auto vr : valRow) {
      row2Rank[vr.row] = vr.rank;
    }
    return row2Rank;
  }
};

#endif
