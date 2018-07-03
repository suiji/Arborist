#include "bagBridge.h"
#include "train.h"
#include "bv.h"

BagBridge::BagBridge(unsigned int nRow_, unsigned int nTree_) :
  nRow(nRow_),
  nTree(nTree_),
  rowBytes(BitMatrix::strideBytes(nRow)),
  raw(RawVector(nTree * rowBytes)) {
}


BagBridge::BagBridge(unsigned int nRow_, unsigned int nTree_, const RawVector &raw_) :
  nRow(nRow_),
  nTree(nTree_),
  rowBytes(BitMatrix::strideBytes(nRow)),
  raw(raw_),
  bmRaw(raw.length() > 0 ? make_unique<BitMatrix>((unsigned int*) &raw[0], nTree, nRow) : make_unique<BitMatrix>(0, 0)) {
}


BagBridge::~BagBridge() {
}


void BagBridge::trainChunk(const Train *train, unsigned int chunkOff) {
  train->getBag((unsigned char*) &raw[chunkOff * rowBytes]);
}


List BagBridge::Wrap() {
  BEGIN_RCPP
  return List::create(
                      _["raw"] = move(raw),
                      _["nRow"] = nRow,
                      _["rowBytes"] = rowBytes,
                      _["nTree"] = nTree
                      );

  END_RCPP
}


unique_ptr<BagBridge> BagBridge::Unwrap(const List &sTrain) {
  List sBag((SEXP) sTrain["bag"]);
  return make_unique<BagBridge>(as<unsigned int>(sBag["nRow"]),
                                as<unsigned int>(sBag["nTree"]),
                                RawVector((SEXP) sBag["raw"]));
}


const BitMatrix *BagBridge::getRaw() {
  return bmRaw.get();
}
