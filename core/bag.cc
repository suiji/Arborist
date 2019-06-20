#include "bag.h"


Bag::Bag(unsigned int* raw,
         unsigned int nTree_,
         unsigned int nRow_) :
  nTree(nTree_),
  nRow(nRow_),
  bitMatrix(make_unique<BitMatrix>(raw, nTree, nRow)) {
}

Bag::Bag() :
  nTree(0),
  nRow(0),
  bitMatrix(make_unique<BitMatrix>(0, 0)) {
}

BitMatrix* Bag::getBitMatrix() const {
  return bitMatrix.get();
}
