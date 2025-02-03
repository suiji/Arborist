
#include "corebridge.h"
#include "fecore.h"

void CoreBridge::init(unsigned nThread) {
  FECore::init(nThread);
}

unsigned int CoreBridge::setNThread(unsigned int nThread) {
  FECore::setNThread(nThread);
  return FECore::getNThread();
}


void CoreBridge::deInit() {
  FECore::deInit();
}
