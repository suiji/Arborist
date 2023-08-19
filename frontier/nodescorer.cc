// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file nodeScorer.cc

   @brief Paramtrized assignment of node scores at frontier.

   @author Mark Seligman
 */

#include "nodescorer.h"
#include "frontier.h"

#include "prng.h"
#include <algorithm>


unique_ptr<NodeScorer> NodeScorer::makeMean() {
  return make_unique<NodeScorer>(&NodeScorer::scoreMean);
}


unique_ptr<NodeScorer> NodeScorer::makePlurality() {
  return make_unique<NodeScorer>(&NodeScorer::scorePlurality);
}

unique_ptr<NodeScorer> NodeScorer::makeLogOdds() {
  return make_unique<NodeScorer>(&NodeScorer::scoreLogOdds);
}


void NodeScorer::frontierPreamble(const Frontier* frontier) {
  ctgJitter = vector<double>(PRNG::rUnif(frontier->getNCtg() * frontier->getNSplit(), 0.5));
}


NodeScorer::NodeScorer(double (NodeScorer::* scorer_)(const SampleMap&,
								  const IndexSet&) const) :
  scorer(scorer_) {
}


double NodeScorer::scoreMean(const SampleMap& smNonterm,
				 const IndexSet& iSet) const {
  return iSet.getSum() / iSet.getSCount();
}


double NodeScorer::scorePlurality(const SampleMap& smNonterm,
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


double NodeScorer::scoreLogOdds(const SampleMap& smNonterm,
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
