// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file rlecresc.cc

   @brief Methods for representing data frame using run-length encoding.

   @author Mark Seligman
 */

#include "rlecresc.h"


RLECresc::RLECresc(size_t nRow_,
                   unsigned int nPredNum,
                   unsigned int nPredFac) :
  nRow(nRow_),
  cardinality(vector<unsigned int>(nPredFac)),
  valOff(vector<size_t>(nPredNum)),
  numVal(vector<double>(0)) {
}


void RLECresc::numSparse(const double feValNum[],
                         const unsigned int feRowStart[],
                         const unsigned int feRunLength[]) {
  size_t colOff = 0;
  for (auto & offset : valOff) {
    offset = numVal.size();
    colOff += numSortSparse(&feValNum[colOff], &feRowStart[colOff], &feRunLength[colOff]);
  }
}


size_t RLECresc::numSortSparse(const double feColNum[],
			       const unsigned int feRowStart[],
			       const unsigned int feRunLength[]) {
  vector<RLEVal<double> > rleNum;
  size_t rleIdx = 0;
  for (size_t rowTot = 0; rowTot < nRow; rowTot += feRunLength[rleIdx++]) {
    rleNum.emplace_back(RLEVal<double>(feColNum[rleIdx], feRowStart[rleIdx], feRunLength[rleIdx]));
  }
  // Postcondition:  rleNum.size() == caller's vector length.

  sort(rleNum.begin(), rleNum.end(), RLECompare<double>);
  encode(rleNum);

  return rleNum.size();
}


void RLECresc::encode(const vector<RLEVal<double> >& rleNum) {
  size_t rowNext = nRow; // Inattainable row number.
  size_t rk = 0;
  numVal.push_back(rleNum[0].val);
  for (auto elt : rleNum) {
    double valThis = elt.val;
    auto rowThis = elt.row;
    auto runCount = elt.extent;
    if (valThis == numVal.back() && rowThis == rowNext) { // Run continues.
      rle.back().extent += runCount;
    }
    else { // New RLE, rank entries regardless whether tied.
      if (valThis != numVal.back()) {
	rk++;
	numVal.push_back(valThis);
      }
      rle.emplace_back(RLEVal<unsigned int>(rk, rowThis, runCount));
    }
    rowNext = rle.back().row + rle.back().extent;
  }
  rleHeight.push_back(rle.size());
}


void RLECresc::numDense(const double feVal[]) {
  unsigned int valIdx = 0;
  for (auto & offset : valOff) {
    offset = numVal.size();

    ValRank<double> valRank(&feVal[valIdx++ * nRow], nRow);
    encode(valRank, numVal);
  }
}


template<typename tn>
void RLECresc::encode(ValRank<tn>& vr, vector<tn>& val, bool valUnique) {
  size_t rowNext = nRow; // Inattainable row number.

  tn valPrev = vr.getVal(0); // Ensures intial rle pushed at first iteration.
  if (valUnique) { // Ensures initial value pushed at first iteration.
    val.push_back(valPrev);
  }
  for (size_t idx = 0; idx < nRow; idx++) {
    auto rowThis = vr.getRow(idx);
    auto valThis = vr.getVal(idx);
    if (valThis == valPrev && rowThis == rowNext) {
      rle.back().extent++;
    }
    else {
      if (!valUnique || valThis != valPrev) {
        val.push_back(valThis);
      }
      rle.emplace_back(RLEVal<unsigned int>(vr.getRank(idx), rowThis, 1));
    }
    valPrev = valThis;
    rowNext = rowThis + 1;
  }
  rleHeight.push_back(rle.size());
}


void RLECresc::facDense(const unsigned int feFac[]) {
  unsigned int facIdx = 0;
  for (auto & card : cardinality) {
    ValRank<unsigned int> valRank(&feFac[facIdx++ * nRow], nRow);

    // Actual factor values are assigned to the 'rank' vector,
    // while a dummy collects the true ranks.
    vector<unsigned int> dummy;
    encode(valRank, dummy, false);
    card = 1 + valRank.getVal(nRow - 1);
  }
}


void RLECresc::dump(vector<size_t>& valOut,
		    vector<size_t>& extentOut,
		    vector<size_t>& rowOut) const {
  size_t i = 0;
  for (auto rlEnc : rle) {
    valOut[i] = rlEnc.val;
    extentOut[i] = rlEnc.extent;
    rowOut[i] = rlEnc.row;
    i++;
  }
}



void RLECresc::dumpRaw(unsigned char rleRaw[]) const {
  for (size_t i = 0; i < getRLEBytes(); i++) {
    rleRaw[i] = ((unsigned char*) &rle[0])[i];
  }
}
