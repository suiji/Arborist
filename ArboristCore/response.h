/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#ifndef ARBORIST_RESPONSE_H
#define ARBORIST_RESPONSE_H

class Response {
  virtual void Nodes(int&) = 0;
 public:
  double *y;
  static nRow; // Set from Predictor
  static Response *response;
  static void Factory(double *yNum);
  static int Factory(const int _yCtg[], double perturb[]);
  static void DeFactory();
  Response(double *_y);
  static void NodeFactory(int &auxSize);
  virtual void PredictOOB(int *conf, double error[]) = 0;
  virtual void GetYRanked(double yRanked[]) = 0;
  virtual ~Response(){}
};

class ResponseReg : public Response {
 public:
  ResponseReg(double *_y);
  ~ResponseReg();
  static int *row2Rank;
  static double *ySorted;
  void Nodes(int& auxSize);
  void PredictOOB(int *conf, double[]);
  void GetYRanked(double yRanked[]);
};

class ResponseCtg : public Response {
  static const int heightLimitWS = 1 << 24; // N.B.:  Revise with more adaptable scheme.
  static int ctgWidth;
  static int leafWSHeight; // Workspace height:  initial estimate comes from first tree.
  static double *leafWS; // Worskpace buffer.
  //  void ExtendWorkspace(int leafCount);
  static double *treeJitter; // Helps prevent ties among response scores.
 public:
  static void Factory(const int _yCtg[], double perturb[], int &_ctgWidth);
  double *FactorFreq(const int _yCtg[], double perturb[]);
  int *yCtg; // The original factor-valued response.
  ResponseCtg(const int _yCtg[], double perturb[], int &_ctgWidth);
  static void ProduceScores(const int *sample2Accum, const class SampleCtg sample[]);
  static double Jitter(int row);
  void Nodes(int &auxSize);
  void GetYRanked(double yRanked[]);
  void PredictOOB(int *conf, double[]);
  ~ResponseCtg();
};
// Run of instances of a given row obtained from sampling for
// an individual tree.
//
class Sample {
 public:
  double val; // Sum of values selected:  rowRun * y-value.
  // Integer-sized container is likely overkill.  Size is typically << #rows,
  // although sample weighting might yield run sizes approaching #rows.
  unsigned int rowRun;
};

// Localizes categorical response.
// Size of 'ctg' container is less than # of rows and may actually
// be quite small.
//
class SampleCtg : public Sample {
 public:
  unsigned int ctg;
};

#endif
