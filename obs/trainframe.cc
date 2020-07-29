// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file trainframe.cc

   @brief Methods associated with ranked frame representations.

   @author Mark Seligman
 */


#include "trainframe.h"
#include "layout.h"
#include "coproc.h"

#include <algorithm>


TrainFrame::TrainFrame(const RLEFrame* rleFrame_,
		       double autoCompress,
		       bool enableCoproc,
		       vector<string>& diag) :
  rleFrame(rleFrame_),
  nRow(rleFrame->nRow),
  coproc(Coproc::Factory(enableCoproc, diag)),
  nPredNum(rleFrame->getNPredNum()),
  cardinality(cardinalities()),
  nPredFac(rleFrame->getNPredFac()),
  nPred(nPredFac + nPredNum),
  predMap(mapPredictors(rleFrame->predForm)) {
  layout = make_unique<Layout>(this, autoCompress);
}


TrainFrame::~TrainFrame() {
}


vector<PredictorT> TrainFrame::cardinalities() const {
  vector<PredictorT> cardPred;
  for (auto facRanked : rleFrame->facRanked) {
    cardPred.push_back(facRanked.size());
  }
  return cardPred;
}


vector<PredictorT> TrainFrame::mapPredictors(const vector<PredictorForm>& predForm_) const {
  vector<PredictorT> core2FE(nPred);
  PredictorT predIdx = 0;
  PredictorT facIdx = nPredNum;
  PredictorT numIdx = 0;
  for (auto form : predForm_) {
    if (form == PredictorForm::factor) {
      core2FE[facIdx++] = predIdx++;
    }
    else {
      core2FE[numIdx++] = predIdx++;
    }
  }
  return core2FE;
}


void TrainFrame::obsLayout() const {
  layout->accumOffsets();
}


IndexT TrainFrame::getDenseRank(PredictorT predIdx) const {
  return layout->getDenseRank(predIdx);
}
