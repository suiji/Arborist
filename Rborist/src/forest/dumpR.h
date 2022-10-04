// Copyright (C)  2019-2022  Mark Seligman
//
// This file is part of rf.
//
// rf is free software: you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 2 of the License, or
// (at your option) any later version.
//
// rf is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with rfR.  If not, see <http://www.gnu.org/licenses/>.

/**
   @file dumpRf.h

   @brief C++ class definitions for managing single-tree forest dump.

   @author Mark Seligman
 */


#ifndef RF_DUMP_RF_H
#define RF_DUMP_RF_H

#include <vector>
#include <memory>
using namespace std;

#include <Rcpp.h>
using namespace Rcpp;

RcppExport SEXP Dump(SEXP sTrain);

struct DumpRf {
  List rfExport;
  List treeOut;

  const IntegerVector predMap;
  const unique_ptr<class ForestExport> forest;
  const IntegerVector factorMap;
  const List facLevel;
  const int factorBase;
  const List treeReg;
  const List leafReg;

  const IntegerVector treePred;
  const IntegerVector leafIdx;
  const IntegerVector delIdx;
  const NumericVector split;
  const IntegerVector cutSense;
  const vector<unsigned char> facBits;
  const NumericVector score;

  static constexpr unsigned int slotBits = 8 * sizeof(unsigned int);
  IntegerVector predInv; // Inversion of predMap.

  stringstream outStr;
  DumpRf(SEXP sArbOut);

  /**
     @brief Dumps tree label and splitting predictor.
   */
  void dumpHead(unsigned int treeIdx);


  /**
     @brief Dumps branch targets of split as C-style ternary.
   */
  void dumpBranch(unsigned int treeIdx);
  
  unsigned int branchTrue(unsigned int treeIdx) const;

  
  unsigned int branchFalse(unsigned int treeIdx) const;


  /**
     @brief Reads split encoding as offset into bit vector.

     @return bit offset associated with tree index.
   */
  size_t getBitOffset(unsigned int treeIdx) const;

  
  /**
     @return cardinality of factor associated with split.
   */
  unsigned int getCardinality(unsigned int treeIdx) const;

  
  void dumpTree();


  void dumpNonterminal(unsigned int treeIdx);


  void dumpNumericSplit(unsigned int treeIdx);


  void dumpFactorSplit(unsigned int treeIdx);

  
  void dumpTerminal(unsigned int treeIdx);
};

#endif
