// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#ifndef SPLIT_RESIDUAL_H
#define SPLIT_RESIDUAL_H

/**
   @file residual.h

   @brief Accumulator class managing implicit quantities as residuals.

   @author Mark Seligman

 */

#include "typeparam.h"
#include <vector>


/**
   @brief Encapsulates imputed residual values.
 */
struct Residual {
  const double sum;  // Imputed response sum over dense indices.
  const IndexT sCount; // Imputed sample count over dense indices.


  /**
     @brief Empty construtor.
   */
  Residual() : sum(0.0), sCount(0) {
  }

  ~Residual() {
  }

  
  /**
     @brief Constructor initializes contents to residual values.

     @param sumExpl is the sum of explicit responses over the cell.

     @param sCountExpl is the sum of explicit sample counts over the cell.
   */
  Residual(double sum_,
           IndexT sCount_);


  bool isEmpty() const {
    return sCount == 0;
  }
  
  /**
     @brief Applies residual to left-moving splitting state.

     @param[out] ySum outputs the residual response sum.

     @param[out] sCount outputs the residual sample count.
   */  
  void apply(FltVal& ySum,
             IndexT& sCount) {
    ySum = this->sum;
    sCount = this->sCount;
  }
};


struct ResidualCtg : public Residual {
  const vector<double> ctgImpl; // Imputed response sums, by category.

  ResidualCtg(double sum_,
              IndexT sCount_,
              const vector<double>& ctgExpl);

  /**
     @brief Empty constructor.
   */
  ResidualCtg() : Residual() {
  }

  ~ResidualCtg() {
  }
  
  /**
     @brief Applies residual to left-moving splitting state.
   */
  void apply(FltVal& ySum,
             IndexT& sCount,
             double& ssR,
             double& ssL,
             class AccumCartCtg* np);
};

#endif
