// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file fecore.cc

   @brief Bridge entry to static initializations.

   @author Mark Seligman
*/

#include "fecore.h"
#include "ompthread.h"


void FECore::init(unsigned int nThread) {
  setNThread(nThread);
}


void FECore::setNThread(unsigned int nThread) {
  OmpThread::setNThread(nThread);
}


unsigned int FECore::getNThread() {
  return OmpThread::getNThread();
}


void FECore::deInit() {
  OmpThread::deInit();
}
