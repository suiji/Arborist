/**
  @file callback.cc

  @brief Implements sorting and sampling utitlities. Employs pre-allocated copy-out parameters to avoid dependence on front end's memory allocation. The core does not implement the callback.h and callback.cc so I have to implement them here...

  @author GitHub user @fyears
 */
#include <Python.h>
#include <numpy/arrayobject.h>
#include <algorithm> // sort
#include <random> // default_random_engine
#include <utility> // make_pair
#include <iostream>
//#include <vector> // vector

#include "callback.h"

unsigned int CallBack::nRow = 0;
bool CallBack::withRepl = false;
double* CallBack::weight;

/**
  @brief Initializes static state parameters for row sampling.

  @param _nRow is the (fixed) number of response rows.

  @param _weight is the user-specified weighting of row samples.

  @param _repl is true iff sampling with replacement.

  @return void.
 */
void CallBack::SampleInit(unsigned int _nRow, double _weight[], bool _repl) {
  nRow = _nRow;
  weight = new double[_nRow];
  std::copy(_weight, _weight+_nRow, weight);
  withRepl = _repl;
  return;
}


int sampleInNumpy(double* weight, unsigned int nRow, bool withRepl, unsigned int nSamp, int out[]){
  // excellent example:
  // http://codereview.stackexchange.com/questions/92266/sending-a-c-array-to-python-numpy-and-back/92353#92353

  Py_Initialize();
  import_array();

  PyObject* pNumpyModule = PyImport_ImportModule("numpy");
  PyObject* pRandomModule = PyObject_GetAttrString(pNumpyModule, "random");
  PyObject* pChoiceModule = PyObject_GetAttrString(pRandomModule , "choice");
  assert(pChoiceModule != NULL);

  npy_intp dims[1] {nRow};

  PyObject* numpyArray = PyArray_Arange(0, nRow, 1, NPY_UINT);
  if (!PyArray_Check(numpyArray)) {
    std::cerr << "Init array failed." << std::endl;
  }
  assert(PyArray_SHAPE((PyArrayObject*)numpyArray)[0] == nRow);

  PyObject* howmany = PyLong_FromLong(nSamp);
  assert(howmany != NULL);

  PyObject* repl = Py_False;
  if (withRepl){
    repl = Py_True;
  }
  assert(repl != NULL);

  PyObject* probs = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, (void *)(weight));
  PyArrayObject* weightSum = (PyArrayObject *) PyArray_Sum((PyArrayObject *)probs, 0, NPY_DOUBLE, NULL);
  probs = PyNumber_TrueDivide(probs, (PyObject *) weightSum);
  if (!PyArray_Check(probs)) {
    std::cerr << "Init weights failed." << std::endl;
  }

  PyObject* pResult = PyObject_CallFunctionObjArgs(pChoiceModule, numpyArray, howmany, repl, probs, NULL);
  assert(pResult != NULL);
  if (!PyArray_Check(pResult)) {
    std::cerr << "Sampling failed." << std::endl;
  }

  PyArrayObject *pArrayResult = (PyArrayObject*)(pResult);
  assert(PyArray_SHAPE(pArrayResult)[0] == nSamp);

  unsigned int* result = (unsigned int*)(PyArray_DATA(pArrayResult));
  for (unsigned int i = 0; i < nSamp; ++i){
    out[i] = result[i];
  }

  //Py_DecRef(pArrayResult);
  Py_DecRef(pResult);
  Py_DecRef(probs);
  Py_DecRef((PyObject *)weightSum);
  Py_DecRef(repl);
  Py_DecRef(howmany);
  Py_DecRef(numpyArray);
  Py_DecRef(pChoiceModule);
  Py_DecRef(pRandomModule);
  Py_DecRef(pNumpyModule);
  Py_Finalize();
}


/**
  @brief Call-back to row sampling.

  @param nSamp is the number of samples to draw.

  @param out[] outputs the sampled row indices.

  @return Formally void, with copy-out parameter vector.
*/
void CallBack::SampleRows(unsigned int nSamp, int out[]) {
  //TODO unable to avoid python here...
  sampleInNumpy(weight, nRow, withRepl, nSamp, out);  
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
  for (int i = one; i <= nRow; ++i)
  {
    pairs.push_back(std::make_pair(ySorted[i-1], rank2Row[i-1]));
  }

  std::sort(pairs.begin(), pairs.end(),
    [](const std::pair<int, int> &a, const std::pair<int, int> &b){
      return a.first < b.first;
    }
  );

  for (int i = one; i <= nRow; ++i) {
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
  for (int i = one; i <= nRow; ++i)
  {
    pairs.push_back(std::make_pair(ySorted[i-1], rank2Row[i-1]));
  }

  std::sort(pairs.begin(), pairs.end(),
    [](const std::pair<double, int> &a, const std::pair<double, int> &b){
      //TODO how to do the double comparation?!
      return a.first < b.first;
    }
  );

  for (int i = one; i <= nRow; ++i) {
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
  std::default_random_engine generator;
  std::uniform_real_distribution<double> distribution(0.0, 1.0);
  for (int i = 0; i < len; i++){
    out[i] = distribution(generator);
  }
}