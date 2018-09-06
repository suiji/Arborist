// Copyright (C)  2012-2018   Mark Seligman
//
// This file is part of ArboristBridgeR.
//
// ArboristBridgeR is free software: you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 2 of the License, or
// (at your option) any later version.
//
// ArboristBridgeR is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with ArboristBridgeR.  If not, see <http://www.gnu.org/licenses/>.

/**
   @file bagBridge.h

   @brief C++ interface to R entry for bagged rows.  There is no direct
   counterpart in Core, which records bagged rows using a bit matrix.

   @author Mark Seligman
 */

#include <Rcpp.h>
using namespace Rcpp;
using namespace std;

class BagBridge {
  size_t nRow;
  unsigned int nTree;
  size_t rowBytes;
  RawVector raw; // Allocated OTF and moved.
  unique_ptr<class BitMatrix> bmRaw;

 public:
  BagBridge(unsigned int nRow_, unsigned int nTree_);
  BagBridge(unsigned int nRow_, unsigned int nTree_, const RawVector &raw_);
  ~BagBridge();

  const size_t getNRow() const {
    return nRow;
  }

  const unsigned int getNTree() const {
    return nTree;
  }

  /**
     @brief Consumes a chunk of tree bags following training.

     @param train is the trained object.

     @param chunkOff is the offset of the current chunk.
   */
  void consume(const class Train* train,
               unsigned int chunkOff);
  List wrap();
  static unique_ptr<BagBridge> unwrap(const List &sBag);
  const BitMatrix* getRaw();
};
