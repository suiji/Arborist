// Copyright (C)  2012-2019   Mark Seligman
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

/**
   @brief Summary of bagged rows, by tree.
 */
class BagBridge {
  const size_t nRow; // # rows trained.
  const unsigned int nTree; // # trees trained.
  const size_t rowBytes; // # count of raw bytes in summary object.
  RawVector raw; // Allocated OTF and moved.
  unique_ptr<class BitMatrix> bmRaw; // Core instantiation of raw data.

 public:
  BagBridge(unsigned int nRow_, unsigned int nTree_);
  BagBridge(unsigned int nRow_, unsigned int nTree_, const RawVector &raw_);
  ~BagBridge();

  /**
     @brief Getter for row count.
   */
  const size_t getNRow() const {
    return nRow;
  }


  /**
     @brief Getter for tree count.
   */
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

  /**
     @brief Bundles trained bag into format suitable for front end.

     @return list containing raw data and summary information.
   */
  List wrap();

  /**
     @brief Read bundled bag information in front-end format.

     @return instantiation containing baga raw data.
   */
  static unique_ptr<BagBridge> unwrap(const List &sBag);

  /**
     @brief Getter for raw data pointer.
   */
  const BitMatrix* getRaw();
};
