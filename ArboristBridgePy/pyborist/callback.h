/**
  @file callback.h

  @brief The core does not implement the callback.h and callback.cc so I have to implement them here...

  @author GitHub user @fyears
 */

#ifndef ARBORIST_CALLBACK_H
#define ARBORIST_CALLBACK_H

#include <vector>

class CallBack {
  static unsigned int nRow;
  static bool withRepl;
  static double* weight;

  public:
    static void SampleInit(unsigned int _nRow,
      double _sampleWeight[],
      bool _withRepl);

    static void SampleRows(unsigned int nSamp,
      int out[]);

    static void QSortI(int ySorted[],
      int rank2Row[],
      int one,
      int nRow);

    static void QSortD(double ySorted[],
      int rank2Row[],
      int one,
      int nRow);

    static void RUnif(int len,
      double out[]);
};

#endif
