/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#ifndef ARBORIST_DATAORD_H
#define ARBORIST_DATAORD_H

class Dord {
  int rank; // True rank, with ties identically receiving lowest applicable value.
  int row; // local copy of r2r[] value.
};


class DataOrd {
  static int *sCountRow;
  static double *sYRow;
  static void CountRows(const int[]);
  static void OrderByRank(const double[], const int[], Dord[]);
 public:
  static void PredByRank(const int predIdx, const class Sample sample[], class SampleOrd predTree[]);
  static void PredByRank(const int predIdx, const class SampleCtg sampleCtg[], class SampleOrdCtg predTreeCtg[]);
  static void SampleRows(const int[],  class Sample sample[], int sample2Rank[], int &bagCount);
  static void SampleRows(const int[], const int[], class SampleCtg sample[], int&);
  static bool *inBag; // Overwritten by each tree.
  static int *sIdxRow; // Inverted by FacResponse for local use.
  static void Factory();
  static Dord *dOrd;
  DataOrd(double*);
  virtual ~DataOrd();
  static void DeFactory();
};

#endif
