/**
  @file callback.cc

  @brief Implements sorting and sampling utitlities. Employs pre-allocated copy-out parameters to avoid dependence on front end's memory allocation. The core does not implement the callback.h and callback.cc so I have to implement them here...

  @author GitHub user @fyears
 */
#include <algorithm> // sort
#include <random> // default_random_engine
#include <utility> // make_pair
//#include <iostream>
//#include <vector> // vector


#include "callback.h"

unsigned int CallBack::nRow = 0;
bool CallBack::withRepl = false;
std::vector<double> CallBack::weight;

/**
  @brief Initializes static state parameters for row sampling.

  @param _nRow is the (fixed) number of response rows.

  @param _weight is the user-specified weighting of row samples.

  @param _repl is true iff sampling with replacement.

  @return void.
 */
void CallBack::SampleInit(unsigned int _nRow, double _weight[], bool _repl) {
  nRow = _nRow;
  weight.assign(_weight, _weight+_nRow);
  withRepl = _repl;
  return;
}


/**
  @brief Call-back to row sampling.

  @param nSamp is the number of samples to draw.

  @param out[] outputs the sampled row indices.

  @return Formally void, with copy-out parameter vector.
*/
void CallBack::SampleRows(unsigned int nSamp, int out[]) {
  std::random_device rd;
  std::mt19937 gen(rd());
  if (withRepl) {
    std::discrete_distribution<unsigned int> distribution(weight.begin(), weight.end());
    for (auto i = 0; i < nSamp; i++){
      out[i] = distribution(gen);
    }
  } else {
    // no replacement
    // so we need another vector to note down the item have been selected or not;
    // we do not ensure/check nSamp <= nRow here
    std::vector<double> w;
    w.assign(weight.begin(), weight.end());
    for (auto i = 0; i < nSamp; ++i)
    {
      std::discrete_distribution<unsigned int> distribution(w.begin(), w.end());
      out[i] = distribution(gen);
      w[out[i]] = 0;
    }
  }
  
}


/**
  @brief Call-back to integer quicksort with indices.

  @param ySorted[] is a copy-out vector containing the sorted integers.

  @param rank2Row[] is the vector of permuted indices.

  @param one is a hard-coded integer indicating unit stride.

  @param nRow is the number of rows to sort.

  @return Formally void, with copy-out parameter vectors.
*/
void CallBack::QSortI(int ySorted[], int rank2Row[], int one, int nRow) {
  std::vector<std::pair<int, int>> pairs;
  for (auto i = one; i <= nRow; ++i)
  {
    pairs.push_back(std::make_pair(ySorted[i-1], rank2Row[i-1]));
  }

  std::sort(pairs.begin(), pairs.end(),
    [](const std::pair<int, int> &a, const std::pair<int, int> &b){
      return a.first < b.first;
    }
  );

  for (auto i = one; i <= nRow; ++i) {
    ySorted[i-1] = pairs[i-1].first;
    rank2Row[i-1] = pairs[i-1].second;
  }
}


/**
  @brief Call-back to double quicksort with indices.

  @param ySorted[] is the copy-out vector of sorted values.

  @param rank2Row[] is the copy-out vector of permuted indices.

  @param one is a hard-coded integer indicating unit stride.

  @param nRow is the number of rows to sort.

  @return Formally void, with copy-out parameter vectors.
*/
void CallBack::QSortD(double ySorted[], int rank2Row[], int one, int nRow) {
  std::vector<std::pair<double, int>> pairs;
  for (auto i = one; i <= nRow; ++i)
  {
    pairs.push_back(std::make_pair(ySorted[i-1], rank2Row[i-1]));
  }

  std::sort(pairs.begin(), pairs.end(),
    [](const std::pair<double, int> &a, const std::pair<double, int> &b){
      //TODO how to do the double comparation?!
      return a.first < b.first;
    }
  );

  for (auto i = one; i <= nRow; ++i) {
    ySorted[i-1] = pairs[i-1].first;
    rank2Row[i-1] = pairs[i-1].second;
  }
}


/**
  @brief Call-back to uniform random-variate generator.

  @param len is number of variates to generate.

  @param out[] is the copy-out vector of generated variates.

  @return Formally void, with copy-out parameter vector.
    
 */
void CallBack::RUnif(int len, double out[]) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> distribution(0.0, 1.0);
  for (auto i = 0; i < len; i++){
    out[i] = distribution(gen);
  }
}
