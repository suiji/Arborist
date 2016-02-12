// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file leaf.h

   @brief Class definitions for sample-oriented aspects of training.

   @author Mark Seligman
 */

#ifndef ARBORIST_LEAF_H
#define ARBORIST_LEAF_H

#include "sample.h"
#include <vector>


class Leaf {
  const class Forest *forest;
 public:
  Leaf(const class Forest *_forest);
  void Extent(const unsigned int frontierMap[], unsigned int bagCount, int tIdx);
  unsigned int TreeHeight(int tIdx);
  void ScoreAccum(int tIdx, int leafIdx, double sum);
  void ScoreReg(int tIdx, int leafIdx, unsigned int _sCount);
  void LeafAccum(int tIdx, unsigned int leafIdx);
  bool Nonterminal(int tIdx, unsigned int off);
  int *ExtentPosition(int tIdx);
  void ScoreCtg(int tIdx, unsigned int off, unsigned int ctg, double wt);
};


class LeafReg : public Leaf {
  std::vector<unsigned int> &sCount;
  std::vector<unsigned int> &rank;
  void Scores(const class SampleReg *sampleReg, const unsigned int frontierMap[], int tIdx);

 public:
  LeafReg(const class Forest *_forest, std::vector<unsigned int> &_sCount, std::vector<unsigned int> &_rank, int bagEst);
  void Leaves(const class SampleReg *sampleReg, const unsigned int frontierMap[], int tIdx);
};


class LeafCtg : public Leaf {
  std::vector<double> &weight;
  unsigned int ctgWidth;
  void Scores(const SampleCtg *sampleCtg, const unsigned int frontierMap[], int tIdx);
 public:
  LeafCtg(const class Forest *_forest, std::vector<double> &_weight, unsigned int _ctgWdith, int heightEst);
  void Leaves(const SampleCtg *sampleCtg, const unsigned int frontierMap[], int tIdx);
};


#endif
