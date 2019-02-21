// This file is part of ArboristCore.

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
   @file rowrank.h

   @brief Class definitions for maintenance of predictor ordering.

   @author Mark Seligman
 */

#ifndef ARBORIST_ROWRANK_H
#define ARBORIST_ROWRANK_H

#include <vector>
#include <tuple>
#include <cmath>

#include "typeparam.h"


typedef tuple<double, unsigned int, unsigned int> NumRLE;
typedef pair<double, unsigned int> ValRowD;
typedef pair<unsigned int, unsigned int> ValRowI;


class RRNode {
 protected:
  unsigned int row;
  unsigned int rank;

 public:
  unsigned int Lookup(unsigned int &_rank) const {
    _rank = rank;
    return row;
  }

  void Init(unsigned int _row, unsigned int _rank) {
    row = _row;
    rank = _rank;
  }

  inline auto getRow() const {
    return row;
  }


  inline auto getRank() const {
    return rank;
  }
};


/**
  @brief Rank orderings of predictors.

*/
class RowRank {
  const unsigned int nRow;
  const unsigned int nPred;
  const unsigned int noRank; // Inattainable rank value.
  unsigned int nPredDense;
  vector<unsigned int> denseIdx;

  // Jagged array holding numerical predictor values for split assignment.
  //const unsigned int *numOffset; // Per-predictor starting offsets.
  //const double *numVal; // Actual predictor values.

  unsigned int nonCompact;  // Total count of uncompactified predictors.
  unsigned int accumCompact;  // Sum of compactified lengths.
  vector<unsigned int> denseRank;
  vector<unsigned int> explicitCount; // Per predictor
  vector<unsigned int> rrStart;   // Predictor offset within rrNode[].
  vector<unsigned int> safeOffset; // Predictor offset within SamplePred[].
  const double autoCompress; // Threshold percentage for autocompression.

  static void Rank2Row(const vector<ValRowD> &valRow,
		       vector<unsigned int> &rowOut,
		       vector<unsigned int> &rankOut);


  /**
     @brief Looks up run characteristics at a given index.

     @param rleIdx is the index.

     @param[out] row outputs the row number.

     @param[out] rank outputs the rank

     @return run length.
   */
  static inline unsigned int runSlot(const unsigned int feRLE[],
				     const unsigned int feRow[],
				     const unsigned int feRank[],
				     unsigned int rleIdx,
				     unsigned int &row,
				     unsigned int &rank) {
    row = feRow[rleIdx];
    rank = feRank[rleIdx];
    return feRLE[rleIdx];
  };

  
  static inline unsigned int runSlot(const unsigned int feRLE[],
				     const unsigned int feRank[],
				     unsigned int rleIdx,
				     unsigned int &rank) {
    rank = feRank[rleIdx];
    return feRLE[rleIdx];
  };

  
  /**
     @brief Walks the design matrix as RLE entries, merging adjacent
     entries with identical ranks.

     @param feRLE are the run lengths corresponding to RLE entries.

     @param rleLength is the count of RLE entries.

     @return total count of explicit slots.
  */
  unsigned int denseBlock(const unsigned int feRank[],
			  const unsigned int feRLE[],
			  unsigned int feRLELength);

  /**
     @brief Determines whether predictor to be stored densely and updates
     storage accumulators accordingly.

     @param predIdx is the predictor under consideration.

     @param denseMax is the highest run length encountered for the predictor:
     must lie within [1, nRow].

     @param argMax is an argmax rank value corresponding to denseMax.

     @return void.
  */
  unsigned int denseMode(unsigned int predIdx,
			 unsigned int denseMax,
			 unsigned int argMax);

  /**
     @brief Assigns predictor offsets according to storage mode:
     noncompressed predictors stored first, as with staging offsets.

     @return void.
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
  void decompress(const unsigned int feRow[],
		  const unsigned int feRank[],
		  const unsigned int feRLE[],
		  unsigned int feRLELength);

 protected:
  vector<RRNode> rrNode; // Row/rank pairs associated with explicit items.
  
 public:

  // Factory parametrized by coprocessor state.
  static RowRank *Factory(const class Coproc *coproc,
			  const class FrameTrain *frameTrain,
			  const unsigned int feRow[],
			  const unsigned int feRank[],
			  const unsigned int feRLE[],
			  unsigned int feRLELength,
			  double _autCompress);

  virtual unique_ptr<class SamplePred> SamplePredFactory(unsigned int bagCount) const;

  virtual unique_ptr<class SPReg> SPRegFactory(
			       const class FrameTrain *frameTrain,
			       unsigned int bagCount) const;

  virtual unique_ptr<class SPCtg> SPCtgFactory(const class FrameTrain *frameTrain,
				    unsigned int bagCount,
				    unsigned int _nCtg) const; 

  RowRank(const class FrameTrain *frameTrain,
	  const unsigned int feRow[],
	  const unsigned int feRank[],
	  const unsigned int feRLE[],
	  unsigned int feRLELength,
	  double _autoCompress);
  virtual ~RowRank();


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


  const RRNode* predStart(unsigned int predIdx) const {
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
};


/**
   @brief Ephemeral proto-RowRank for presorting.  Builds copyable vectors
   characterizing both a RowRank and an accompanying numerical BlockSparse.
 */
class RankedPre {
  unsigned int nRow;
  unsigned int nPredNum;
  unsigned int nPredFac;

  // To be consumed by front-end variant of RowRank.
  vector<unsigned int> rank;
  vector<unsigned int> row;
  vector<unsigned int> runLength;

  // To be consumed by front-end variant of BlockSparse.
  vector<unsigned int> numOff;
  vector<double> numVal;

  unsigned int numSortSparse(const double feColNum[],
			     const unsigned int feRowStart[],
			     const unsigned int feRunLength[]);

  void rankNum(const vector<NumRLE> &rleNum);

  void numSortRaw(const double colNum[]);

  /**
     @brief Stores ordered predictor column, entering uncompressed.

     @param valRow contains the value/row-number pairs.

     @return void.
  */
  void rankNum(const vector<ValRowD> &valRow);

  /**
     @brief Sorts factors and stores as rank-ordered run-length encoding.

     @param predCol is a single predictor's observations, by row.

     @return void.
  */
  void facSort(const unsigned int predCol[]);

  /**
     @brief Builds rank-ordered run-length encoding to hold factor values.

     Final "rank" values are the internal factor codes and may contain
     gaps.  A dense numbering scheme would entail backmapping at LH bit
     assignment following splitting (q.v.):  prediction and training
     must map to the same factor levels.

     @return void.
  */ 
  void rankFac(const vector<ValRowI> &valRow);
  
  
 public:

  RankedPre(unsigned int _nRow,
	    unsigned int _nPredNum,
	    unsigned int _nPredFac);

  /**
     @brief Accessor for copyable rank vector.
   */
  const vector<unsigned int> &Rank() const {
    return rank;
  }

  /**
     @brief Accessor for copyable row vector.
   */
  const vector<unsigned int> &Row() const {
    return row;
  }

  /**
     @brief Accessor for copyable run-length vector.
   */
  const vector<unsigned int> &RunLength() const {
    return runLength;
  }


  /** 
      @brief Accessor for copyable offset vector.
   */
  const vector<unsigned int> &NumOff() const {
    return numOff;
  }

  /**
     @brief Accessor for copyable numerical value vector.
   */
  const vector<double> &NumVal() const {
    return numVal;
  }
  

  /**
     @brief Presorts runlength-encoded numerical block suppled by front end.

     @param feValNum[] is a vector of numerical values.

     @param feRowStart[] maps row indices to offset within value vector.

     @param feRunLength[] is length of each run of values.

     @return void.
   */
  void numSparse(const double feValNum[],
		 const unsigned int feRowStart[],
		 const unsigned int feRunLength[]);


  /**
     @brief Presorts dense numerical block supplied by front end.
   */
  void numDense(const double feNum[]);

  
  /**
     @brief Presorts dense factor block supplied by front end.
   */
  void facDense(const unsigned int feFac[]);
};





/**
   @brief Sparse predictor-ranked numerical block.
 */
class BlockRanked {
  const double *val;
  const unsigned int *offset;

  /**
     @return rank of specified predictor at specified rank.
   */
  inline double RankVal(unsigned int predIdx,
		       unsigned int rk) const {
    return val[offset[predIdx] + rk];
  }


 public:
  BlockRanked(const double _val[],
	      const unsigned int _offset[]) :
  val(_val),
    offset(_offset) {
    }


  /**
     @brief Derives split values for a numerical predictor by synthesizing
     a fractional intermediate rank and interpolating.

     @param predIdx is the predictor index.

     @param rankRange is the range of ranks.

     @return interpolated predictor value at synthesized rank.
  */
  inline double QuantRank(unsigned int predIdx,
			  RankRange rankRange,
			  const vector<double> &splitQuant) const {
    double rankNum = rankRange.rankLow + splitQuant[predIdx] * (rankRange.rankHigh - rankRange.rankLow);
    unsigned int rankFloor = floor(rankNum);
    unsigned int rankCeil = ceil(rankNum);

    return RankVal(predIdx, rankFloor) + (rankNum - rankFloor) * (RankVal(predIdx, rankCeil) - RankVal(predIdx, rankFloor));
 }
};


/**
   @brief Front end-created container caching preformatted summary of
   training data.
 */
class RankedSet {
  const RowRank *rowRank;
  const BlockRanked *numRanked;

 public:
  RankedSet(const RowRank *_rowRank,
	     const BlockRanked *_numRanked) :
  rowRank(_rowRank),
  numRanked(_numRanked) {
  }


  /**
     @brief Getter for rowRank.
   */
  const RowRank *getRowRank() const {
    return rowRank;
  }

  /**
     @brief Getter for numRanked.
   */
  const BlockRanked *getNumRanked() const {
    return numRanked;
  }
};

#endif

