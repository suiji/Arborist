/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include "predictor.h"
#include <R.h>

// Testing only:
#include <iostream>
using namespace std;

// Establishes the layout of the predictors relative to their container arrays.
// The container arrays themselves are allocated, as well, if cloning is specified.
//
int Predictor::nPred = 0;
int Predictor::nRow = 0;
int Predictor::nPredNum = 0;
int Predictor::nPredInt = 0;
int Predictor::nPredFac = 0;

double *Predictor::predProb = 0;
double* Predictor::numBase = 0;
int* Predictor::facBase = 0;
int* Predictor::intBase = 0;
int* Predictor::facWidth = 0;
int* Predictor::facSum = 0;
int  Predictor::facTot = 0; // Total count of factor levels.
int  Predictor::maxFacWidth = 0;

bool Predictor::numClone = false;
bool Predictor::intClone = false;
bool Predictor::facClone = false;


// The block routines, below, together act as "subfactories" for
// the complete Predictor object.
// Numeric predictors.
//
void Predictor::NumericBlock(double xn[], int _nrow, int _ncol, bool doClone) {
  Factory(_nrow);
  numClone = doClone; // Signals desctructor call at end.
  nPredNum = _ncol;
  nPred += nPredNum;

  if (doClone) { // Data subject to alteration:  clone.
    int bufSize = _nrow * _ncol;
    numBase = new double[bufSize];
    for (int i = 0; i < bufSize; i++)
      numBase[i] = xn[i];
  }
  else // Data will not be altered nor resides in R temporary:  can cache R's copy.
    numBase = xn;
}

// Ordered integer factors.
//
void Predictor::IntegerBlock(int xi[], int nrow, int ncol, bool doClone) {
  Factory(nrow);
  intClone = doClone;
  nPredInt = ncol;
  nPred += nPredInt;

  if (doClone) { // Data subject to alteration:  clone.
    int bufSize = nrow * ncol;
    intBase = new int[bufSize];
    for (int i = 0; i < bufSize; i++) {
      intBase[i] = xi[i];
    }
  }
  else // Data will not be altered nor resides in R temporary:  can cache R's copy.
    intBase = xi;
}

// Unordered integer factors.  Space is allocated on the fly, so cloning is
// unnecessary.
//
// Factor blocks are subject to alteration and come from R temporaries, so
// are always cloned.
//
void Predictor::FactorBlock(int xi[], int _nrow, int _ncol, int levelCount[]) {
  Factory(_nrow);
  facClone = true;
  nPredFac = _ncol;
  nPred += nPredFac;

  int bufSize = _ncol * _nrow;
  facBase = new int[bufSize];
  for (int i = 0; i < bufSize; i++)
    facBase[i] = xi[i] - 1; // Not necessary to zero-justify, but hey.

  facSum = new int[nPredFac];
  facWidth = new int[nPredFac];

  int facOff = 0;
  maxFacWidth = 0;
  for (int i = 0; i < nPredFac; i++) {
    facSum[i] = facOff;
    facWidth[i] = levelCount[i];
    if (facWidth[i] > maxFacWidth)
      maxFacWidth = facWidth[i];
    facOff += levelCount[i];
  }
  facTot = facOff;
}

// N.B. Must be called after # predictors finalized.
//
void Predictor::SetProbabilities(const double _predProb[]) {
  predProb = new double[nPred];
  for (int i = 0; i < nPred; i++)
    predProb[i] = _predProb[i];
}

void Predictor::Factory(int _nRow) {
  if (nRow > 0 && nRow != _nRow)
    cout << "Conflicting predictor training values" << endl;
  nRow = _nRow;
}

void Predictor::DeFactory() {
  nPred = nPredNum = nPredInt = nPredFac = facTot = nRow = 0;
  delete [] predProb;
  predProb = 0;
  if (facWidth)
    delete [] facWidth;
  facWidth = 0;
  if (facSum)
    delete [] facSum;
  facSum = 0;
  if (numClone)
    delete [] numBase;
  if (intClone)
    delete [] intBase;
  if (facClone) {
    delete [] facBase;
  }
  numBase = 0;
  facBase = 0;
  intBase = 0;
  facClone = false;
  numClone = false;
  intClone = false;
}

// Derives a vector of ranks via wasteful sorting.  Original predictors should
// be cloned before using.
// Implemented here to avoid exposing "iterator" outside of class.
//
void Predictor::UniqueRank(int *rank2Row) {
  int predIdx;

  int baseOff = 0;
  int rankOff = 0;
  for (predIdx = 0; predIdx < nPredNum; predIdx++, baseOff += nRow, rankOff += nRow){
    /* Sort-with-index requires a vector of rows to permute.*/
    for (int i = 0; i < nRow; i++)
      *(rank2Row + rankOff + i) = i;
    /*   rsort_with_index(numBase + predIdx * nRow, rank2Row + predIdx*nRow, nRow);
    */

    // TODO:  Replace with thread-safe sort to permit parallel execution.
    // Row consistency does not appear necessary, so an unstable sort probably
    // suffices:
    R_qsort_I(numBase + baseOff, rank2Row + rankOff, 1, nRow);
  }

  // Note divergence of 'baseOff' and 'rankOff':
  baseOff = 0;
  for (predIdx = 0; predIdx < nPredFac; predIdx++, baseOff += nRow, rankOff += nRow) {
    for (int i = 0; i < nRow;i++)
      *(rank2Row + rankOff + i) = i;
    // TODO:  Replace with thread-safe sort to permit parallel execution.
    R_qsort_int_I(facBase + baseOff, rank2Row + rankOff, 1, nRow);
  }
}

// rank2row is only read twice per predictor, early on, but reused by each tree.
//
// Orders 'x', columnwise, according to the ranking of the elements of predictor.
// All indices within a column must be used so that all elements of 'y' are present.
// Ties, then, must be handled by a method which uses all available indices, such as
// the "first" or "random" methods used by the rank() function.
//
//  row2Rank = apply(x, 2, rank, ties.method="first");  
//
// Implemented here to avoid exposing iterator outside of class.
//
void Predictor::SetSortAndTies(const int* rank2Row, Dord *dOrd) {
  int baseOff = 0;
  int rankOff = 0;
  int predIdx;

  //#pragma omp parallel default(shared) private(predIdx)//, baseOff, rankOff)
  {
    //  #pragma omp for schedule(static, 1) nowait
  for (predIdx = 0; predIdx < nPredNum; predIdx++, baseOff += nRow, rankOff += nRow) {
      OrderByRank(numBase + baseOff, rank2Row + rankOff, dOrd + rankOff);
    }

  }
  baseOff = 0;
  //#pragma omp parallel default(shared) private(predIdx)//, baseOff, rankOff)
  {    // Factors:
    //#pragma omp for schedule(static, 1) nowait
    for (predIdx = 0; predIdx < nPredFac; predIdx++, baseOff += nRow, rankOff += nRow) {
      OrderByRank(facBase + baseOff, rank2Row + rankOff, dOrd + rankOff);
    }
  }
}


// Orders predictor data by rank into 'dCol' vector.  Data of interest includes the row index and tie class.
// Row index is obtained directly from r2r[].  Tie class derived by comparing 'x' values of consecutive ranks.
//
void Predictor::OrderByRank(const double xCol[], const int r2r[], Dord dCol[]) {
  // Sorts the rows of 'y' in the order that this predictor increases.
  // Sorts the predictor for later identification of tie classes.
  int row =r2r[0];
  Dord dLoc;
  int ord = 0;
 
  dLoc.rank = ord; // Can use 'rk', provided splits are built using ranks i/o rows.
  dLoc.row = row;
  dCol[0] = dLoc;
  double prevX = xCol[0];
  for (int rk = 1; rk < nRow; rk++) {
    row = r2r[rk];
    double curX = xCol[rk]; // Can be looked up by rank if dummy sorted x values saved by caller.
    ord = curX == prevX ? ord : rk;// Numeric case requires distinct, but indexable, 'rk'.
    dLoc.rank = ord;
    dLoc.row = row;
    dCol[rk] = dLoc;
    prevX = curX;
  }
}
// Orders predictor data by rank into 'dCol' vector.  Data of interest includes the row index and tie class.
// Row index is obtained directly from r2r[].  Tie class derived by comparing 'x' values of consecutive ranks.
//
void Predictor::OrderByRank(const int xCol[], const int r2r[], Dord dCol[], bool ordinals) {
  // Sorts the rows of 'y' in the order that this predictor increases.
  // Sorts the predictor for later identification of tie classes.
  int row =r2r[0];
  Dord dLoc;
  int ord = 0;
  dLoc.rank = ord;
  dLoc.row = row;
  dCol[0] = dLoc;
  int prevX = xCol[0];
  for (int rk = 1; rk < nRow; rk++) {
    row = r2r[rk];
    int curX = xCol[rk]; // Can be looked up by rank if dummy sorted x values saved by caller.
    ord = curX == prevX ? ord : (ordinals ? ++ord : rk);// Integer case uses 'rk' as index; factors require actual ordinals.
    dLoc.rank = ord;
    dLoc.row = row;
    dCol[rk] = dLoc;
    prevX = curX;
  }
}
