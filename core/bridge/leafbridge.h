// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file leafbridge.h

   @brief Front-end wrappers for core Leaf objects.

   @author Mark Seligman
 */

#ifndef CORE_BRIDGE_LEAFBRIDGE_H
#define CORE_BRIDGE_LEAFBRIDGE_H

struct LeafBridge {
  /**
     @brief Getter for number of rows under prediction.
   */
  size_t getRowPredict() const;

  virtual ~LeafBridge() {
  }


  virtual class LeafFrame* getLeaf() const = 0;
};


struct LeafRegBridge : public LeafBridge {

  LeafRegBridge(const unsigned int* height,
                unsigned int nTree,
                const unsigned char* node,
                const unsigned int* bagHeight,
                const unsigned char* bagSample,
                const double* yTrain,
                size_t rowTrain,
                double trainMean,
                size_t predictRow);

  ~LeafRegBridge();


  void dump(const struct BagBridge* bagBridge,
            vector<vector<size_t> >& rowTree,
            vector<vector<unsigned int> >& sCountTree,
            vector<vector<double> >& scoreTree,
            vector<vector<unsigned int> >& extentTree) const;


  /**
     @brief Pass-through to core method.
   */
  const vector<double>& getYPred() const;

  class LeafFrame* getLeaf() const;

private:
  unique_ptr<class LeafFrameReg> leaf;
};


struct LeafCtgBridge : public LeafBridge {

  LeafCtgBridge(const unsigned int* height,
                unsigned int nTree,
                const unsigned char* node,
                const unsigned int* bagHeight,
                const unsigned char* bagSample,
                const double* weight,
                unsigned int ctgTrain,
                size_t rowPredict,
                bool doProb);

  ~LeafCtgBridge();

  
  /**
     @brief Dumps bagging and leaf information into per-tree vectors.
   */
  void dump(const struct BagBridge* bagBridge,
            vector<vector<size_t> > &rowTree,
            vector<vector<unsigned int> > &sCountTree,
            vector<vector<double> > &scoreTree,
            vector<vector<unsigned int> > &extentTree,
            vector<vector<double> > &_probTree) const;

  class LeafFrame* getLeaf() const;

  void vote();

  const unsigned int* getCensus() const;
  
  const vector<unsigned int>& getYPred() const;

  unsigned int getCtgTrain() const;

  const vector<double>& getProb() const;
  
  unsigned int getYPred(size_t row) const;
  
  unsigned int ctgIdx(unsigned int ctgTest,
                      unsigned int ctgPred) const;
  
private:
  unique_ptr<class LeafFrameCtg> leaf;
};

#endif
