``Pyborist``
============

This is the sub-project of the python wrapper of ``Arborist`` C++ library for Random Forest (TM) algorithm.

The project is under heavy development.


Installation
------------

To compile this project, a platform-specific C++ compiler (``msvc`` on Windows, ``gcc`` / ``g++`` on Linux, ``clang`` / ``clang++`` on OS X) is needed, as well as Python of course. ``numpy`` and ``cython`` are also needed.

.. code-block:: bash

    pip install numpy
    pip install 'cython>=0.24'

    git clone https://github.com/fyears/Arborist.git
    cd Arborist/ArboristBridgePy/
    python setup.py install


Example
-------

You could use this simple example to check whether the package is installed correctly:

.. code-block:: python

    from sklearn import datasets
    from sklearn.cross_validation import train_test_split
    from sklearn.metrics import log_loss

    from pyborist import PyboristClassifier, PyboristRegressor

    iris = datasets.load_iris().data
    response = datasets.load_iris().target

    X_train, X_test, y_train, y_test = train_test_split(iris, response)

    k = PyboristClassifier(n_estimators=30)
    k.fit(X_train, y_train)
    log_loss(y_test, k.predict_proba(X_test))

    # 0.12783...


Development
-----------

Project Structure
~~~~~~~~~~~~~~~~~

Currently:

.. code-block:: text

    .
    ├── LICENSE
    ├── Makefile
    ├── pyborist
    │   ├── callback.cc
    │   ├── callback.h
    │   ├── cy*.pxd
    │   ├── cy*.pyx
    │   ├── ...
    │   ├── __init__.py
    │   └── skl.py
    ├── README.rst
    ├── requirenments.txt
    └── setup.py


Coding Style
~~~~~~~~~~~~

The coding style for ``pyborist`` might be a little confusing: it mixes different styles for a reason.

The simple policy:

Every ``*.pyx`` and ``*.pxd`` files that import and directly wrap the corresponding C / C++ functions / classes follow the coding styles and naming conventions in the ``Arborist`` core source code. For example, use ``theVariableName`` and ``ClassName`` if possible. Moreover, the names and filenames of the imported classes should reflex the corresponding original ones, like: ``Train`` in ``train.h`` and ``train.cc`` -> ``Train`` and ``PyTrain`` in ``cytrain.pxd`` and ``cytrain.pyx``.

Every pure Python files follows the ``scikit-learn`` coding styles. For example, ``the_variable_name`` is more favorable. And the parameters should try to follow their counterparts in package ``scikit-learn``.


License
-------

Currently this Python-bridge is released under `MIT License <https://opensource.org/licenses/MIT>`_. Please notice that the core source code and the R wrapper source code may be released under other license(s).
