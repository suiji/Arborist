// Copyright (C)  2019 - 2024   Mark Seligman and Decision Patterns LLC.
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
   @file exprdump.cc

   @brief C++ interface to R entry for expression dumper.

   @author Mark Seligman
 */

#include "trainR.h"
#include "forestR.h"
#include "exprdumpR.h"


// [[Rcpp::export]]
RcppExport SEXP exprdump(SEXP sArbOut) {

  ExprDump dumper(sArbOut);
  return dumper.exprTree();
}


ExprDump::ExprDump(SEXP sArbOut) :
  primExport((SEXP) expandTrainRcpp(sArbOut)),
  treeOut((SEXP) primExport["tree"]),
  predNames((SEXP) primExport["predNames"]),
  predMap((SEXP) primExport["predMap"]),
  tree(ForestExpand::unwrap(List(sArbOut), predMap)),
  factorMap((SEXP) primExport["factorMap"]),
  factorLevel((SEXP) primExport["factorLevel"]),
  factorBase(predMap.length() - factorMap.length()),
  treeReg((SEXP) treeOut["internal"]),
  treePred((SEXP) treeReg["predIdx"]),
  leafIdx((SEXP) treeReg["leafIdx"]),
  delIdx((SEXP) treeReg["delIdx"]),
  split((SEXP) treeReg["split"]),
  cutSense((SEXP) treeReg["invert"]),
  facBits(tree.getFacSplitTree(0)),
  leafReg((SEXP) treeOut["leaf"]),
  score((SEXP) leafReg["score"]) {
}


List ExprDump::exprTree() const {
  List exprList;
  unsigned int treeIdx = 0;
  while (treeIdx < delIdx.length()) {
    exprList.push_back(exprBlock(treeIdx));
  }

  return exprList;
}


ExpressionVector ExprDump::exprBlock(unsigned int& treeIdx) const {
  ExpressionVector exprVec;
  for (; delIdx[treeIdx] != 0; treeIdx++) {
    exprVec.push_back(nonterminal(treeIdx)[0]);
  }
  exprVec.attr("value") = getTerminalValue(treeIdx++);

  return exprVec;
}


ExpressionVector ExprDump::nonterminal(unsigned int treeIdx) const {
  return getPredictor(treeIdx) < factorBase ? numericSplit(treeIdx) : factorSplit(treeIdx);
}


ExpressionVector ExprDump::numericSplit(unsigned int treeIdx) const {
  stringstream ss;
  // True:  peel, out of box.  False:  support, next in box.
  ss << getPredictorName(treeIdx) << cutString(treeIdx) << split[treeIdx];

  return ExpressionVector(ss.str());
}


string ExprDump::cutString(unsigned int treeIdx) const {
  return cutSense[treeIdx] == 1 ? " > " : " < ";
}


ExpressionVector ExprDump::factorSplit(unsigned int treeIdx) const {
  unsigned int predIdx = getPredictor(treeIdx);
  size_t bitOffset = getBitOffset(treeIdx);
  stringstream ss;

  // Factor values for true (peel) branch:  invert to " != " (implicit &&).
  ss << getPredictorName(treeIdx) << " %in% c(";
  bool prefixComma = false;
  for (unsigned int fac = 0; fac < getCardinality(predIdx); fac++) {
    if (!levelPeels(bitOffset + fac)) {
      ss << (prefixComma ? ", " : "") << getLevelName(predIdx, fac);
      prefixComma = true;
    }
  }
  ss << ")";
  return ExpressionVector(ss.str());
}


bool ExprDump::levelPeels(size_t bit) const {
  return facBits[bit / slotBits] & (1ul << (bit & (slotBits - 1)));
}


size_t ExprDump::getBitOffset(unsigned int treeIdx) const {
  return *((unsigned int*) &split[treeIdx]);
}


string ExprDump::getLevelName(unsigned int predIdx, unsigned int fac) const {
  stringstream ss;
  IntegerVector factors((SEXP) factorLevel[predIdx - factorBase]);
  StringVector factorNames(factors.attr("levels"));
  ss << StringVector(CharacterVector((SEXP) factorNames[fac]));
  return ss.str();
}


unsigned int ExprDump::getCardinality(unsigned int predIdx) const {
  IntegerVector factors((SEXP) factorLevel[predIdx - factorBase]);
  StringVector factorNames(factors.attr("levels"));
  return factorNames.length();
}


string ExprDump::getPredictorName(unsigned int treeIdx) const {
  unsigned int predIdx = getPredictor(treeIdx);
  unsigned int predUser = predMap[predIdx];
  stringstream ss;
  ss << predNames[predUser];
  return ss.str();
}


double ExprDump::getTerminalValue(unsigned int treeIdx) const {
  // "value" attribute of exprvec.
  // TODO:  Range should be verified in constructor.
  return score[leafIdx[treeIdx]];
}
