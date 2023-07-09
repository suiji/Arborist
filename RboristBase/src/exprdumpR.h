// Copyright (C)  2019-2023  Mark Seligman and Decision Patterns LLC.
//
// This file is part of RboristBase.
//
// RboristBase is free software: you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 2 of the License, or
// (at your option) any later version.
//
// RboristBase is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with RboristBase.  If not, see <http://www.gnu.org/licenses/>.

/**
   @file exprPrim.h

   @brief C++ class definitions for managing single-tree expression dump.

   @author Mark Seligman
 */


#ifndef RBORISTCORE_EXPRDUMP_H
#define RBORISTCORE_EXPRDUMP_H

#include <vector>
#include <memory>
using namespace std;

#include <Rcpp.h>
using namespace Rcpp;

#include "forestR.h"


RcppExport SEXP exprdump(SEXP sTrain);


struct ExprDump {
  List primExport;
  List treeOut;

  const StringVector predNames; // Internally-index predictor names.
  const IntegerVector predMap;
  const ForestExpand tree;
  const IntegerVector factorMap;
  const List factorLevel; // List of factor (integer) vectors.
  const unsigned int factorBase; // Compressed index into factor level.
  const List treeReg;

  const IntegerVector treePred;
  const IntegerVector leafIdx;
  const IntegerVector delIdx;
  const NumericVector split;
  const IntegerVector cutSense;
  const vector<unsigned char> facBits;
  const List leafReg;
  const NumericVector score;

  static constexpr unsigned int slotBits = 8 * sizeof(unsigned int);
  
  stringstream outStr;

  ExprDump(SEXP sArbOut);

  /**
     @brief Exprs tree label and splitting predictor.
   */
  void exprHead(unsigned int treeIdx);


  /**
     @brief Exprs branch targets of split as C-style ternary.
   */
  void exprBranch(unsigned int treeIdx);
  
  unsigned int branchTrue(unsigned int treeIdx) const;

  
  unsigned int branchFalse(unsigned int treeIdx) const;


  /**
     @brief Reads split encoding as offset into bit vector.

     @return bit offset associated with tree index.
   */
  size_t getBitOffset(unsigned int treeIdx) const;


  /**
     @brief Determines whether a given factor level is peeled.

     @param bit is the bit position representing the level's value.

     @return true iff factor-level is to be peeled.
   */
  bool levelPeels(size_t bit) const;
  
  /**
     @return cardinality of factor associated with split.
   */
  unsigned int getCardinality(unsigned int treeIdx) const;


  unsigned int getPredictor(unsigned int treeIdx) const {
    return static_cast<unsigned int>(treePred[treeIdx]);
  }
  
  
  List exprTree() const;


  ExpressionVector exprBlock(unsigned int& blockHead) const;
  

  ExpressionVector nonterminal(unsigned int treeIdx) const;


  ExpressionVector numericSplit(unsigned int treeIdx) const;


  ExpressionVector factorSplit(unsigned int treeIdx) const;


  /**
     @return leaf score assocated with terminal.
   */
  double getTerminalValue(unsigned int treeIdx) const;

  string getPredictorName(unsigned int treeIdx) const;


  /**
     @return string encoding opposite sense of cut.
   */
  string cutString(unsigned int treeIdx) const;
  

  string getLevelName(unsigned int treeIdx,
		      unsigned int fac) const;
};

#endif
