#include "bag.h"


Bag::Bag(unsigned int* raw,
         unsigned int nTree_,
         size_t nObs_) :
  nTree(nTree_),
  nObs(nObs_),
  bitMatrix(make_unique<BitMatrix>(raw, nTree, nObs)) {
}

Bag::Bag() :
  nTree(0),
  nObs(0),
  bitMatrix(make_unique<BitMatrix>(0, 0)) {
}

BitMatrix* Bag::getBitMatrix() const {
  return bitMatrix.get();
}
