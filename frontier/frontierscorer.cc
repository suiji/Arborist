// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file frontierscorer.cc

   @brief Paramtrized assignment of node scores at frontier.

   @author Mark Seligman
 */

#include "frontierscorer.h"
#include "frontier.h"

#include "prng.h"
#include <algorithm>


unique_ptr<FrontierScorer> FrontierScorer::makeMean() {
  return make_unique<FrontierScorer>(&FrontierScorer::scoreMean);
}


unique_ptr<FrontierScorer> FrontierScorer::makePlurality() {
  return make_unique<FrontierScorer>(&FrontierScorer::scorePlurality);
}

unique_ptr<FrontierScorer> FrontierScorer::makeLogOdds() {
  return make_unique<FrontierScorer>(&FrontierScorer::scoreLogOdds);
}


void FrontierScorer::frontierPreamble(const Frontier* frontier) {
  ctgJitter = vector<double>(PRNG::rUnif(frontier->getNCtg() * frontier->getNSplit(), 0.5));
}


FrontierScorer::FrontierScorer(double (FrontierScorer::* scorer_)(const SampleMap&,
								  const IndexSet&) const) :
  scorer(scorer_) {
}


double FrontierScorer::scoreMean(const SampleMap& smNonterm,
				 const IndexSet& iSet) const {
  return iSet.getSum() / iSet.getSCount();
}


double FrontierScorer::scorePlurality(const SampleMap& smNonterm,
				    const IndexSet& iSet)  const {
  const double* nodeJitter = &ctgJitter[iSet.getCtgSumCount().size() * iSet.getSplitIdx()];
  PredictorT argMax = 0;// TODO:  set to nCtg and error if no count.
  IndexT countMax = 0;
  PredictorT ctg = 0;
  for (const SumCount& sc : iSet.getCtgSumCount()) {
    IndexT sCount = sc.getSCount();
    if (sCount > countMax) {
      countMax = sCount;
      argMax = ctg;
    }
    else if (sCount > 0 && sCount == countMax) {
      if (nodeJitter[ctg] > nodeJitter[argMax]) {
	argMax = ctg;
      }
    }
    ctg++;
  }

  //  argMax, ties broken by jitters, plus its own jitter.
  return argMax + nodeJitter[argMax];
}


double FrontierScorer::scoreLogOdds(const SampleMap& smNonterm,
				    const IndexSet& iSet) const {
  // Walks the sample indices associated with the node index,
  // accumulating a sum of pq-values.
  //
  double pqSum = 0.0;
  IndexRange range = smNonterm.range[iSet.getSplitIdx()];
  for (IndexT idx = range.getStart(); idx != range.getEnd(); idx++) {
    IndexT sIdx = smNonterm.sampleIndex[idx];
    pqSum += gamma[sIdx];
  }

  return iSet.getSum() / pqSum;
}
