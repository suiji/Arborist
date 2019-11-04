// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file rleframe.h

   @brief Methods for representing data frame using run-length encoding.

   @author Mark Seligman
 */


#include "rleframe.h"
#include "valrank.h"

#include <tuple>


RLECresc::RLECresc(size_t nRow_,
                   unsigned int nPredNum,
                   unsigned int nPredFac) :
  nRow(nRow_),
  cardinality(vector<unsigned int>(nPredFac)),
  valOff(vector<unsigned int>(nPredNum)),
  numVal(vector<double>(0)) {
}


void RLECresc::numSparse(const double feValNum[],
                         const unsigned int feRowStart[],
                         const unsigned int feRunLength[]) {
  unsigned int colOff = 0;
  for (auto & num : valOff) {
    num = numVal.size();
    unsigned int idxCol = numSortSparse(&feValNum[colOff], &feRowStart[colOff], &feRunLength[colOff]);
    colOff += idxCol;
  }
}


unsigned int RLECresc::numSortSparse(const double feColNum[],
                                     const unsigned int feRowStart[],
                                     const unsigned int feRunLength[]) {
  vector<RLEVal<double> > rleNum;
  unsigned int rleIdx = 0;
  for (size_t rowTot = 0; rowTot < nRow; rowTot += feRunLength[rleIdx++]) {
    rleNum.emplace_back(RLEVal<double>(feColNum[rleIdx], feRowStart[rleIdx], feRunLength[rleIdx]));
  }
  // Postcondition:  rleNum.size() == caller's vector length.

  sort(rleNum.begin(), rleNum.end(), RLECompare<double>);
  encode(rleNum);

  return rleNum.size();
}


void RLECresc::encode(const vector<RLEVal<double> > &rleNum) {
  RLEVal<double> elt = rleNum[0];
  unsigned int rk = 0;
  numVal.push_back(elt.val);
  rle.emplace_back(RLEVal<unsigned int>(rk, elt.row, elt.runLength));
  for (unsigned int idx = 1; idx < rleNum.size(); idx++) {
    elt = rleNum[idx];
    double valThis = elt.val;
    unsigned int rowThis = elt.row;
    unsigned int runCount = elt.runLength;
    if (valThis == numVal.back() && rowThis == rle.back().row + rle.back().runLength) {
      rle.back().runLength += runCount;
    }
    else { // New RLE, rank entries regardless whether tied.
      if (valThis != numVal.back()) {
	rk++;
	numVal.push_back(valThis);
      }
      rle.emplace_back(RLEVal<unsigned int>(rk, rowThis, runCount));
    }
  }
}


void RLECresc::numDense(const double feVal[]) {
  unsigned int valIdx = 0;
  for (auto & num : valOff) {
    num = numVal.size();

    ValRank<double> valRank(&feVal[valIdx++ * nRow], nRow);
    encode(valRank, numVal);
  }
}


template<typename tn>
void RLECresc::encode(ValRank<tn>& vr, vector<tn>& val, bool valUnique) {
  unsigned int rowThis = vr.getRow(0);
  tn valThis = vr.getVal(0); // Assumes >= 1 row.
  val.push_back(valThis);
  rle.emplace_back(RLEVal<unsigned int>(vr.getRank(0), rowThis, 1));

  for (size_t idx = 1; idx < nRow; idx++) {
    unsigned int rowPrev = rowThis;
    rowThis = vr.getRow(idx);
    tn valPrev = valThis;
    valThis = vr.getVal(idx);
    bool sameVal = valThis == valPrev;
    if (sameVal && rowThis == (rowPrev + 1)) {
      rle.back().runLength++;
    }
    else {
      if (!valUnique || !sameVal) {
        val.push_back(valThis);
      }
      rle.emplace_back(RLEVal<unsigned int>(vr.getRank(idx), rowThis, 1));
    }
  }
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


void RLECresc::dumpRLE(unsigned char rleRaw[]) const {
  for (size_t i = 0; i < getRLEBytes(); i++) {
    rleRaw[i] = ((unsigned char*) &rle[0])[i];
  }
}


RLEFrame::RLEFrame(size_t nRow_,
                   const vector<unsigned int>& cardinality_,
                   size_t rleLength_,
                   const RLEVal<unsigned int>* rle_,
                   unsigned int nPredNum_,
                   const double* numVal_,
                   const unsigned int* valOff_) :
  nRow(nRow_),
  cardinality(cardinality_),
  rleLength(rleLength_),
  rle(rle_),
  nPredNum(nPredNum_),
  numVal(numVal_),
  valOff(valOff_) {
}
