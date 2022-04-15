// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file samplemap.cc

   @brief Groups samples by their terminal assignments.

   @author Mark Seligman
 */


#include "samplemap.h"
#include "sampler.h"


vector<double> SampleMap::row2Score(const Sampler* sampler,
				    unsigned int tIdx,
				    const vector<double>& score) const {
  vector<double> rowScore(sampler->getNObs());
  vector<IndexT> sample2row = sampler->sampledRows(tIdx);
  IndexT leafIdx = 0;
  for (IndexRange leafRange : range) {
    double leafScore = score[ptIdx[leafIdx++]];
    for (IndexT idx = leafRange.getStart(); idx != leafRange.getEnd(); idx++) {
      IndexT sIdx = sampleIndex[idx];
      rowScore[sample2row[sIdx]] = leafScore;
    }
  }
  
  return rowScore;
}
