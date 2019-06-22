// This file is part of framemap.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file rankedframe.h

   @brief Class definitions for maintenance of predictor ordering.

   @author Mark Seligman
 */

#ifndef ARBORIST_RANKEDFRAME_H
#define ARBORIST_RANKEDFRAME_H

#include <vector>
#include <cmath>

#include "rleframe.h"
#include "rowrank.h"

using namespace std;

#include "typeparam.h"

/**
  @brief Rank orderings of predictors.
*/
class RankedFrame {
  const unsigned int nRow;
  const unsigned int nPred;
  const unsigned int noRank; // Inattainable rank value.
  unsigned int nPredDense;
  vector<unsigned int> denseIdx;

  unsigned int nonCompact;  // Total count of uncompactified predictors.
  unsigned int accumCompact;  // Sum of compactified lengths.
  vector<unsigned int> denseRank;
  vector<unsigned int> explicitCount; // Per predictor
  vector<unsigned int> rrStart;   // Predictor offset within rrNode[].
  vector<unsigned int> safeOffset; // Predictor offset within SamplePred[].
  const unsigned int denseThresh; // Threshold run length for autocompression.

  // Move to SummaryFrame:
  vector<unsigned int> cardinality;

  /**
     @brief Walks the design matrix as RLE entries, merging adjacent
     entries with identical ranks.

     @param feRLE are the run lengths corresponding to RLE entries.

     @param rleLength is the count of RLE entries.

     @return total count of explicit slots.
  */
  unsigned int denseBlock(const RLEVal<unsigned int> feRLE[],
			  size_t feRLELength);

  /**
     @brief Determines whether predictor to be stored densely and updates
     storage accumulators accordingly.

     @param predIdx is the predictor under consideration.

     @param denseMax is the highest run length encountered for the predictor:
     must lie within [1, nRow].

     @param argMax is an argmax rank value corresponding to denseMax.
  */
  unsigned int denseMode(unsigned int predIdx,
			 unsigned int denseMax,
			 unsigned int argMax);

  /**
     @brief Assigns predictor offsets according to storage mode:
     noncompressed predictors stored first, as with staging offsets.
  */
  void modeOffsets();

  
  /**
     @brief Decompresses a block of predictors deemed not to be storable
     densely.

     @param feRow[] are the rows corresponding to distinct runlength-
     encoded (RLE) entries.

     @param feRank[] are the ranks corresponing to RLE entries.

     @param feRLE records the run lengths spanning the original design
     matrix.

     @param rleLength is the total count of RLE entries.
  */
  void decompress(const RLEVal<unsigned int> feRLE[],
		  size_t feRLELength);

 protected:
  vector<RowRank> rrNode; // Row/rank pairs associated with explicit items.
  
 public:

  // Factory parametrized by coprocessor state.
  static RankedFrame *Factory(const class Coproc *coproc,
                              unsigned int nRow,
                              const vector<unsigned int>& cardinality,
                              unsigned int nPred,
                              const RLEVal<unsigned int> feRLE[],
                              size_t feRLELength,
                              double autoCompress);

  RankedFrame(unsigned int nRow_,
              const vector<unsigned int>& cardinality,
              const unsigned int nPred,
              const RLEVal<unsigned int> feRLE[],
              size_t feRLELength,
              double autoCompress);

  virtual ~RankedFrame();


  inline unsigned int getNRow() const {
    return nRow;
  }
  
  
  inline unsigned int getNPred() const {
    return nPred;
  }


  inline unsigned int NoRank() const {
    return noRank;
  }

  
  inline unsigned int getExplicitCount(unsigned int predIdx) const {
    return explicitCount[predIdx];
  }


  /**
     @brief Accessor for dense rank value associated with a predictor.

     @param predIdx is the predictor index.

     @return dense rank assignment for predictor.
   */
  unsigned int getDenseRank(unsigned int predIdx) const{
    return denseRank[predIdx];
  }

  
  /**
     @brief Computes a conservative buffer size, allowing strided access
     for noncompact predictors but full-width access for compact predictors.

     @param stride is the desired strided access length.

     @return buffer size conforming to conservative constraints.
   */
  unsigned int safeSize(unsigned int stride) const {
    return nonCompact * stride + accumCompact; // TODO:  align.
  }

  
  /**
     @brief Computes conservative offset for storing predictor-based
     information.

     @param predIdx is the predictor index.

     @param stride is the multiplier for strided access.

     @param extent outputs the number of slots avaiable for staging.

     @return safe offset.
   */
  unsigned int getSafeOffset(unsigned int predIdx,
                          unsigned int stride,
                          unsigned int &extent) const {
    extent = denseRank[predIdx] == noRank ? stride : explicitCount[predIdx];
    return denseRank[predIdx] == noRank ? safeOffset[predIdx] * stride : nonCompact * stride + safeOffset[predIdx]; // TODO:  align.
  }


  const RowRank* predStart(unsigned int predIdx) const {
    return &rrNode[rrStart[predIdx]];
  }


  /**
     @brief Getter for count of dense predictors.

     @return number of dense predictors.
   */
  inline unsigned int getNPredDense() const {
    return nPredDense;
  }


  /**
     @brief Accessor for dense index vector.

     @return reference to vector.
   */
  inline const vector<unsigned int> &getDenseIdx() const {
    return denseIdx;
  }


  inline unsigned int getCardinality(unsigned int facIdx) const {
    return cardinality[facIdx];
  }
};


#endif

