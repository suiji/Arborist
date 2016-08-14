// Copyright (C)  2012-2016   Mark Seligman
//
// This file is part of ArboristBridgeR.
//
// ArboristBridgeR is free software: you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 2 of the License, or
// (at your option) any later version.
//
// ArboristBridgeR is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with ArboristBridgeR.  If not, see <http://www.gnu.org/licenses/>.

/**
   @file callback.h

   @brief Exposes utility functions provided by the front end.

   @author Mark Seligman
 */

#ifndef ARBORIST_CALLBACK_H
#define ARBORIST_CALLBACK_H

class CallBack {
 public:
  static void SampleInit(unsigned int _nRow, const double _sampleWeight[], bool _withRepl);
  static void SampleRows(unsigned int nSamp, int out[]);
  static void RUnif(int len, double out[]);
  static void QSortI(int ySorted[], unsigned int rank2Row[], int one, int nRow);
  static void QSortD(double ySorted[], unsigned int rank2Row[], int one, int nRow);
};

#endif
