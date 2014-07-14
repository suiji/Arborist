/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include "facrun.h"

int *BHeap::vacant = 0;
BHPair *BHeap::bhPair = 0;
FacRun *FacRun::levelFR = 0;
int *FacRun::levelFROrd = 0;
int FacRun::accumCount = 0;
double *FacRunCtg::facCtgSum = 0;
int FacRunCtg::ctgWidth = -1;

void FacRun::Factory(int _accumCount) {
  accumCount = _accumCount;
  int vacCount = accumCount * Predictor::nPredFac;
  BHeap::vacant = new int[vacCount];
  for (int i = 0; i < vacCount; i++)
    BHeap::vacant[i] = 0;

  BHeap::bhPair = new BHPair[accumCount * Predictor::facTot];
  levelFR = new FacRun[accumCount * Predictor::facTot];
  levelFROrd = new int[accumCount * Predictor::facTot];
}

void FacRun::ReFactory(int _accumCount) {
  delete [] BHeap::vacant;
  delete [] BHeap::bhPair;
  delete [] levelFR;
  delete [] levelFROrd;

  Factory(_accumCount);
}

void FacRunCtg::Factory(const int _accumCount, const int _ctgWidth) {
  ctgWidth = _ctgWidth;
  FacRun::Factory(_accumCount);
  facCtgSum = new double[_accumCount * Predictor::facTot * ctgWidth];
}

void FacRunCtg::ReFactory(const int _accumCount) {
  FacRun::ReFactory(_accumCount);

  delete [] facCtgSum;
  facCtgSum = new double[_accumCount * Predictor::facTot * ctgWidth];
}

void FacRunCtg::DeFactory() {
  delete [] facCtgSum;
  facCtgSum = 0;
  ctgWidth = -1;
  FacRun::DeFactory();
}

void FacRun::DeFactory() {
  delete [] BHeap::vacant;
  delete [] BHeap::bhPair;
  delete [] levelFR;
  delete [] levelFROrd;

  BHeap::vacant = 0;
  BHeap::bhPair = 0;
  levelFR = 0;
  levelFROrd = 0;
}
