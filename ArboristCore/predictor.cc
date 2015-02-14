// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file predictor.cc

   @brief Methods for maintaining predictor-specific information.

   @author Mark Seligman
 */

#include "predictor.h"
#include "sample.h"
#include "callback.h"

// Testing only:
//#include <iostream>
//using namespace std;

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
int* Predictor::facCard = 0;
int* Predictor::facSum = 0;
int  Predictor::nCardTot = 0; // Total count of factor levels.
int  Predictor::maxFacCard = -1;

bool Predictor::numClone = false;
bool Predictor::intClone = false;
bool Predictor::facClone = false;


// The block routines, below, together act as "subfactories" for
// the complete Predictor object.


/**
   @brief Copies numeric-valued observations as block, if needed.

   @param xn[] contains the matrix of numeric observations.

   @param _nrow is the number of rows.

   @param _ncol is the number of columns.

   @param doClone indicates whether copying is required.

   @return void.
 */
void Predictor::NumericBlock(double xn[], int _nrow, int _ncol, bool doClone) {
  numClone = doClone; // Signals desctructor call at end.
  nPredNum = _ncol;

  if (doClone) { // Data subject to alteration:  clone.
    int bufSize = _nrow * _ncol;
    numBase = new double[bufSize];
    for (int i = 0; i < bufSize; i++)
      numBase[i] = xn[i];
  }
  else // Data will not be altered nor resides in R temporary:  can cache R's copy.
    numBase = xn;
}

/**
   @brief Copies integer-valued observations as block, if needed.

   @param xi[] contains the matrix of integer observations.

   @param _nrow is the number of rows.

   @param ncol is the number of columns.

   @param doClone indicates whether copying is required.

   @return void.
 */
void Predictor::IntegerBlock(int xi[], int _nrow, int ncol, bool doClone) {
  intClone = doClone;
  nPredInt = ncol;

  if (doClone) { // Data subject to alteration:  clone.
    int bufSize = _nrow * ncol;
    intBase = new int[bufSize];
    for (int i = 0; i < bufSize; i++) {
      intBase[i] = xi[i];
    }
  }
  else // Data will not be altered nor resides in R temporary:  can cache R's copy.
    intBase = xi;
}

/**
   @brief Enumerates and adjusts factor-valued observation block.

   @param xi[] contains the matrix of factor observations.

   @param _nrow is the number of rows.

   @param _ncol is the number of columns.

   @param levelCount enumerates the factor cardinalities.

   @return void.
*/
void Predictor::FactorBlock(int xi[], int _nrow, int _ncol, int levelCount[]) {
  facClone = true;
  nPredFac = _ncol;

  int bufSize = _ncol * _nrow;
  facBase = new int[bufSize];
  for (int i = 0; i < bufSize; i++)
    facBase[i] = xi[i] - 1; // Not necessary to zero-justify, but hey.

  facSum = new int[nPredFac];
  facCard = new int[nPredFac];

  int facOff = 0;
  maxFacCard = 0;
  for (int i = 0; i < nPredFac; i++) {
    facSum[i] = facOff;
    facCard[i] = levelCount[i];
    if (facCard[i] > maxFacCard)
      maxFacCard = facCard[i];
    facOff += levelCount[i];
  }
  nCardTot = facOff;
}

/**
   @brief Light off the initializations needed by the Preditor class.

   @param _predProb is the vector selection probabilities.

   @param _nPred is the number of predictors.

   @param _nRow is the number of observations.

   @return void.
 */
void Predictor::Factory(const double _predProb[], int _nPred, int _nRow) {
  nPred = _nPred;
  nRow = _nRow;
  if (_predProb != 0)  {
    SetProbabilities(_predProb);
  }
}

/**
   @brief Deallocates and resets.

   @return void.
 */
void Predictor::DeFactory() {
  if (predProb != 0) {
    delete [] predProb;
    predProb = 0;
  }

  nPred = nPredNum = nPredInt = nPredFac = nCardTot = nRow = 0;  
  if (facCard)
    delete [] facCard;
  facCard = 0;
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

/**
 @brief Derives a vector of ranks via callback methods.

 @param rank2Row outputs the permutation matrix defined by sorting individual columns.

 @return void, with output vector parameter.
*/
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
    CallBack::QSortD(numBase + baseOff, rank2Row + rankOff, 1, nRow);
  }

  // Note divergence of 'baseOff' and 'rankOff':
  baseOff = 0;
  for (predIdx = 0; predIdx < nPredFac; predIdx++, baseOff += nRow, rankOff += nRow) {
    for (int i = 0; i < nRow;i++)
      *(rank2Row + rankOff + i) = i;
    // TODO:  Replace with thread-safe sort to permit parallel execution.
    CallBack::QSortI(facBase + baseOff, rank2Row + rankOff, 1, nRow);
  }
}

/**
   @brief Establishes predictor orderings used by all trees.

   @param rank2Row is the matrix of permutations defined by per-predictor sorting.

   @param predOrd outputs the matrix of predictor-order objects.

   @return void, with output parameter matrix.
*/
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
void Predictor::SetSortAndTies(const int* rank2Row, PredOrd *predOrd) {
  int baseOff = 0;
  int rankOff = 0;
  int predIdx;

  //#pragma omp parallel default(shared) private(predIdx)//, baseOff, rankOff)
  {
    //  #pragma omp for schedule(static, 1) nowait
  for (predIdx = 0; predIdx < nPredNum; predIdx++, baseOff += nRow, rankOff += nRow) {
      OrderByRank(numBase + baseOff, rank2Row + rankOff, predOrd + rankOff);
    }

  }
  baseOff = 0;
  //#pragma omp parallel default(shared) private(predIdx)//, baseOff, rankOff)
  {    // Factors:
    //#pragma omp for schedule(static, 1) nowait
    for (predIdx = 0; predIdx < nPredFac; predIdx++, baseOff += nRow, rankOff += nRow) {
      OrderByRank(facBase + baseOff, rank2Row + rankOff, predOrd + rankOff);
    }
  }
}

/**
   @brief Encapsulates predictor data by rank, with index and tie class.

   @param r2r is the permutation vector derived from sorting.

   @param dCol outputs the predictor-order object.
   
   @return void, with output vector parameter.
*/
// Row index is obtained directly from r2r[].  Tie class derived by comparing 'x' values of consecutive ranks.
//
void Predictor::OrderByRank(const double xCol[], const int r2r[], PredOrd dCol[]) {
  // Sorts the rows of 'y' in the order that this predictor increases.
  // Sorts the predictor for later identification of tie classes.
  int row =r2r[0];
  PredOrd dLoc;
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

/**
   @brief Same as above, but with option for strict ordinal rank numbering.

   @param r2r is the permutation vector derived from sorting.

   @param dCol outputs the predictor-order object.

   @param ordinals indicates whether rank number is to be ordinal-based.
   
   @return void, with output vector parameter.
*/
void Predictor::OrderByRank(const int xCol[], const int r2r[], PredOrd dCol[], bool ordinals) {
  // Sorts the rows of 'y' in the order that this predictor increases.
  // Sorts the predictor for later identification of tie classes.
  int row =r2r[0];
  PredOrd dLoc;
  int ord = 0;
  dLoc.rank = ord;
  dLoc.row = row;
  dCol[0] = dLoc;
  int prevX = xCol[0];
  for (int rk = 1; rk < nRow; rk++) {
    row = r2r[rk];
    int curX = xCol[rk]; // Can be looked up by rank if dummy sorted x values saved by caller.
    ord = curX == prevX ? ord : (ordinals ? (ord+1) : rk);// Integer case uses 'rk' as index; factors require actual ordinals.
    dLoc.rank = ord;
    dLoc.row = row;
    dCol[rk] = dLoc;
    prevX = curX;
  }
}

/**
 @brief Creates an internal copy of front-end probability vector.

 @param _predProb is a probability vector suppled by the front end.

 @return void.
*/
void Predictor::SetProbabilities(const double _predProb[]) {
  predProb = new double[nPred];

  for (int i = 0; i < nPred; i++) {
    predProb[i] = _predProb[i];
  }
}

/**
   @brief Derives split values for a predictor.

   @param predIdx is the preditor index.

   @param rkLow is the lower rank of the split.

   @param rkHigh is the higher rank of the split.

   @return mean predictor value between low and high ranks.
 */

double Predictor::SplitVal(int predIdx, int rkLow, int rkHigh) {
  double *numCol = numBase + predIdx * nRow;
  double low = numCol[rkLow];
  double high = numCol[rkHigh];

  /* ASSERTIONS:
  if (rkLow < 0 || rkLow > nRow || rkHigh <0 || rkHigh > nRow)
    cout << "NONSENSICAL split" << rkLow << " / " << rkHigh << " : " << predIdx << endl;
  if (rkLow == rkHigh)
    cout << "TRIVIAL SPLIT " << rkLow << " / " << rkHigh << endl;
  else if (low > high)
    cout << "BAD SPLIT  (" << predIdx << ") "<<  low << " / " << high << " ords:  " << rkLow << " / " << rkHigh <<endl;
  else if (low == high)
    cout << "TIED SPLIT:  " << low << " / " << high << endl;
  */
  return 0.5 * (low + high);
}
