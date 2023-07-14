// Copyright (C)  2019 - 2023   Mark Seligman
//
// This file is part of RboristBase.
//
// RboristBase is free software: you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 2 of the License, or
// (at your option) any later version.
//
// You should have received a copy of the GNU General Public License
// along with RboristBase.  If not, see <http://www.gnu.org/licenses/>.

/**
   @file dumpR.cc

   @brief C++ interface to R entry for export methods.

   @author Mark Seligman
 */

#include "trainR.h"
#include "dumpR.h"
#include "forestR.h"
#include "forestbridge.h"

/**
   @brief Structures forest summary for analysis by Dump package.

   @param sForest is the Forest summary.

   @return RboristDump as List.
 */
RcppExport SEXP Dump(SEXP sArbOut) {
  BEGIN_RCPP

  DumpRf dumper(sArbOut);
  dumper.dumpTree();

  return StringVector(dumper.outStr.str());
  END_RCPP
}


DumpRf::DumpRf(SEXP sArbOut) :
  rfExport((SEXP) expandTrainRcpp(sArbOut)),
  treeOut((SEXP) rfExport["tree"]),
  predMap((SEXP) rfExport["predMap"]),
  forest(ForestExpand::unwrap(List(sArbOut), predMap)),
  factorMap((SEXP) rfExport["factorMap"]),
  facLevel((SEXP) rfExport["predFactor"]),
  factorBase(predMap.length() - factorMap.length()),
  treeReg((SEXP) treeOut["internal"]),
  leafReg((SEXP) treeOut["leaf"]),
  treePred((SEXP) treeReg["predIdx"]),
  leafIdx((SEXP) treeReg["leafIdx"]),
  delIdx((SEXP) treeReg["delIdx"]),
  split((SEXP) treeReg["split"]),
  cutSense((SEXP) treeReg["cutSense"]),
  facBits(forest.getFacSplitTree(0)),
  score((SEXP) leafReg["score"]),
  predInv(IntegerVector(predMap.length())) {
  predInv[predMap] = IntegerVector(seq(0, predMap.length() - 1));
}


void DumpRf::dumpTree() {
  for (R_xlen_t treeIdx = 0; treeIdx < delIdx.length(); treeIdx++) {
    delIdx[treeIdx] == 0 ?  dumpTerminal(treeIdx) : dumpNonterminal(treeIdx);
  }
}


void DumpRf::dumpNonterminal(unsigned int treeIdx) {
  if (predInv[treePred[treeIdx]] < factorBase) {
    dumpNumericSplit(treeIdx);
  }
  else {
    dumpFactorSplit(treeIdx);
  }
}


void DumpRf::dumpHead(unsigned int treeIdx) {
  outStr << treeIdx << ":  @" << treePred[treeIdx];
}



void DumpRf::dumpNumericSplit(unsigned int treeIdx) {
  dumpHead(treeIdx);
  outStr << (cutSense[treeIdx] == 1 ? " <= " : " >= ")<< split[treeIdx];
  dumpBranch(treeIdx);
}

void DumpRf::dumpBranch(unsigned int treeIdx) {
  outStr << " ? " << branchTrue(treeIdx) << " : " << branchFalse(treeIdx) << endl;
}


unsigned int DumpRf::branchTrue(unsigned int treeIdx) const {
  return treeIdx + delIdx[treeIdx] + 1;
}


unsigned int DumpRf::branchFalse(unsigned int treeIdx) const {
  return treeIdx + 1;
}


size_t DumpRf::getBitOffset(unsigned int treeIdx) const {
  return *((unsigned int*) &split[treeIdx]);
}


unsigned int DumpRf::getCardinality(unsigned int treeIdx) const {
  unsigned int predIdx = treePred[treeIdx];
  unsigned int facIdx = predInv[predIdx] - factorBase;
  return StringVector((SEXP) facLevel[facIdx]).length();
}


void DumpRf::dumpFactorSplit(unsigned int treeIdx) {
  dumpHead(treeIdx);

  bool first = true;
  size_t bitOffset = getBitOffset(treeIdx);

  outStr << " in {";
  for (unsigned int fac = 0; fac < getCardinality(treeIdx); fac++) {
    size_t bit = bitOffset + fac;
    if (facBits[bit / slotBits] & (1ul << (bit & (slotBits - 1)))) {
      outStr << (first ? "" : ", ") << fac;
      first = false;
    }
  }
  outStr << "}";
  dumpBranch(treeIdx);
}


void DumpRf::dumpTerminal(unsigned int treeIdx) {
  outStr << treeIdx << ":  leaf score ";
  R_xlen_t scoreIdx = leafIdx[treeIdx];
  if (scoreIdx >= delIdx.length()) {
    outStr << " (error) " << endl;
  }
  else {
    outStr << score[scoreIdx]  << endl;
 }
}
