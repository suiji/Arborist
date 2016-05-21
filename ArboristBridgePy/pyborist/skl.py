import numpy as np

from .cyrowrank import PyRowRank
from .cytrain import PyTrain
from .cypredict import PyPredict

__all__ = ['PyboristClassifier', 'PyboristRegressor']



class PyboristModel(object):
    """The scikit-learn API for Pyborist model.

    Parameters
    ----------
    n_estimators: int, optional (default=10)
        The number of trees to train.

    bootstrap: bool, optional (default=True)
        Whether row sampling is by replacement.

    class_weight: str or None or dict-like, optional (default=None)
        Proportional weighting of classification categories.
        Could use None or `balance`, or a dict as {'class_label': weight}

    min_info_ratio: float, optional (default=0.01)
        Information ratio with parent below which node does not split.

    min_samples_split: int, optional (default=2)
        Minimum number of distinct row references to split a node.

    max_depth: int or None, optional (default=0)
        Maximum number of tree levels to train. Zero denotes no limit.

    no_validate: bool, optional (default=True)
        Whether to train without validation.

    n_to_sample: int, optional (default=0)
        Number of rows to sample, per tree.

    max_features: int, optional (default=0)
        Number of trial predictors for a split.

    pred_prob: float, optional (default=0.0)
        Probability of selecting individual predictor as trial splitter.
        Causes each predictor to be selected as a splitting candidate with distribution Bernoulli(pred_prob).

    quantiles_arr: array-like or None (default=None)
        Quantile levels to validate.

    q_bin: int, optional (default=5000)
        Bin size for facilating quantiles at large sample count.

    reg_mono: array-like or None, optional (default=None)
        Signed probability constraint for monotonic regression.

    tree_block: int, optional (default=1)
        Maximum number of trees to train during a single level (e.g., coprocessor computing).

    pvt_block: int, optional (default=8)
        Maximum number of trees to train in a block (e.g., cluster computing).

    is_classifier: bool, optional (default = False)
        Regression or Classification?

    Attributes
    ----------
    n_features_ : int
        The number of features.

    n_samples_ : int
        The number of samples in trainning dataset.

    n_classes_ : int
        The number of classes in classification.

    classes_ : array-like
        The class labels extracted from the training response in classification.

    n_outputs_ : int
        The number of outputs.
    """
    def __init__(self,
        n_estimators = 10,
        bootstrap = True,
        class_weight = None,
        min_info_ratio = 0.01,
        min_samples_split = 2,
        max_depth = 0,
        no_validate = True,
        n_to_sample = 0,
        max_features = 0,
        pred_prob = 0.0,
        quantiles_arr = None,
        q_bin = 5000,
        reg_mono = None,
        tree_block = 1,
        pvt_block = 8,
        is_classifier = False):
        # a trick to save everything into self...
        for k, v in locals().items():
            if k == 'self':
                continue
            setattr(self, k, v)


    @staticmethod
    def get_allowed_param_keys():
        """
        Get the default params for this model.
        """
        keys = ['n_estimators',
            'bootstrap',
            'class_weight',
            'min_info_ratio',
            'min_samples_split',
            'max_depth',
            'no_validate',
            'n_to_sample',
            'max_features',
            'pred_prob',
            'quantiles_arr',
            'q_bin',
            'reg_mono',
            'tree_block',
            'pvt_block',
            'is_classifier']
        return keys


    def fit(self,
        X,
        y,
        sample_weight = None,
        feature_weight = None):
        """Fit estimator.

        Parameters
        ----------
        X : array-like, shape=(n_samples, n_features)
            The input samples.

        y: array-like, shape=(n_samples)
            The input response.

        sample_weight : array-like or None, shape=(n_samples), optional (default=None)
            The sample weight.

        feature_weight : array-like or None, shape=(n_features), optional (default=None)
            The feature weight.

        Returns
        -------
        self : object
            Returns self.
        """
        if X.shape[0] != y.shape[0]:
            raise ValueError('X and y do not share the same first dimension.')

        X = X.astype(np.double, copy=False)
        if self.is_classifier:
            self._estimator_type = 'classifier'
            self.classes_, y = np.unique(y, return_inverse=True)
            self.n_classes_ = len(self.classes_)
            y = y.astype(np.uintc, copy=False)
        else:
            self._estimator_type = 'regressor'
            y = y.astype(np.double, copy=False)

        self.n_samples_ = X.shape[0]
        self.n_features_ = X.shape[1]
        self.real_params = {}

        if sample_weight is None:
            sample_weight = np.ones(self.n_samples_)
        else:
            if len(sample_weight) != self.n_samples_:
                raise ValueError('Sample weight should equal to column number.')
            if np.any(sample_weight < 0.0):
                raise ValueError('Sample weight should larger than zero.')
            if np.sum(sample_weight) == 0.0:
                raise ValueError('Sample weight could not be all zero.')

        if feature_weight is None:
            feature_weight = np.ones(self.n_features_)
        else:
            if len(feature_weight) != self.n_features_:
                raise ValueError('Predictor weight should equal to column number.')
            if np.any(feature_weight < 0.0):
                raise ValueError('Predictor weight should larger than zero.')
            if np.sum(feature_weight) == 0.0:
                raise ValueError('Predictor weight could not be all zero.')

        self._valid_default_params()
        self._init_row_rank(X, y, sample_weight, feature_weight)
        self._adjust_model_params(X, y, sample_weight, feature_weight)

        if self.is_classifier:
            self._train_classification(X, y, sample_weight, feature_weight)
        else:
            self._train_regression(X, y, sample_weight, feature_weight)

        return self


    def _valid_default_params(self):
        """
        Check the default params that are independent of X and y
        """
        if self.min_samples_split <= 0:
            raise ValueError('Invalid min_samples_split.')

        if self.quantiles_arr is not None:
            if np.any(self.quantiles_arr < 0.0) or np.any(self.quantiles_arr > 1.0):
                raise ValueError('Quantiles shoule be inside 0 and 1.')
            if np.any(np.diff(self.quantiles_arr) < 0.0):
                raise ValueError('Quantiles should be increasing.')

        if self.pred_prob < 0.0 or self.pred_prob > 1.0:
            raise ValueError('pred_prob should be inside [0.0, 1.0].')

        if self.is_classifier:
            if self.quantiles_arr is not None:
                raise ValueError('Quantiles are for regression only.')


    def _init_row_rank(self,
        X,
        y,
        sample_weight,
        feature_weight):
        """Call the backend to generate the rowrank. Similar to the R verson.

        Attributes
        ----------
        row: array-like, shape=(n_samples * n_features)

        rank: array-like, shape=(n_samples * n_features)

        inv_num: array-like, shape=(n_samples * n_features)

        Returns
        -------
        self : object
            Returns self.
        """
        n_samples = self.n_samples_
        n_features = self.n_features_

        rank = np.empty([n_samples * n_features], dtype=np.intc)
        row = np.empty([n_samples * n_features], dtype=np.intc)
        inv_num = np.zeros([n_samples * n_features], dtype=np.intc)
        PyRowRank.PreSortNum(np.ascontiguousarray(X.transpose().reshape(X.size)),
            n_features,
            n_samples,
            row,
            rank,
            inv_num)

        self.real_params.update({
            'presort_row': row,
            'presort_rank': rank,
            'presort_inv_num': inv_num
        })
        return self


    def _adjust_model_params(self,
        X,
        y,
        sample_weight,
        feature_weight):
        """Regenerate some parameters of model based on input.

        Returns
        -------
        self : object
            Returns self.
        """
        n_samples = self.n_samples_
        n_features = self.n_features_

        self.real_params.update({
            'n_to_sample': self.n_to_sample
        })
        if self.n_to_sample == 0:
            self.real_params.update({
                'n_to_sample': n_samples \
                    if self.bootstrap \
                    else np.round((1-np.exp(-1)) * n_samples)
            })

        if self.min_samples_split > n_samples:
            raise ValueError('Invalid min_samples_split.')

        if self.is_classifier:
            n_classes = self.n_classes_

            # the accepted class_weight is None or 'balanced' or dict-like
            # we want to storage an array-like, sorted based on the self.classes_

            if self.class_weight is None:
                class_weight_arr = np.ones(n_classes)
            elif self.class_weight == 'balanced':
                class_weight_arr = n_samples / (n_classes * np.bincount(y))
            else:
                if np.any(np.array(list(self.class_weight.values())) < 0.0):
                    raise ValueError('Invalid class weighting.')
                if np.sum(list(self.class_weight.values())) == 0.0:
                    raise ValueError('Invalid class weighting.')
                if len(self.class_weight) < n_classes:
                    raise ValueError('Some class weighting missed.')
                class_weight_arr = np.array([self.class_weight[k] for k in self.classes_])
            class_weight_arr[np.isinf(class_weight_arr)] = 0
            class_weight_arr = class_weight_arr / np.sum(class_weight_arr)
            class_weight_arr = (class_weight_arr[y] + 
                (np.random.uniform(size=n_samples) - 0.5) * 
                0.5 / n_samples / n_samples).astype(np.double)
            self.real_params.update({
                'class_weight': class_weight_arr
            })

            if self.max_features > n_features:
                raise ValueError('max_features should be no more than n_features.')
            max_features = self.max_features
            if self.max_features == 0 and self.pred_prob == 0.0 and n_features < 16:
                max_features = np.floor(np.sqrt(n_features))
            self.real_params.update({
                'max_features': max_features
            })

            pred_prob = self.pred_prob
            if self.pred_prob == 0.0 and self.max_features == 0:
                pred_prob = np.ceil(np.sqrt(n_features)) / n_features
            mean_weight = 1.0 if pred_prob == 0.0 else pred_prob
            prob_arr = feature_weight * (n_features * mean_weight) / np.sum(feature_weight)
            self.real_params.update({
                'prob_arr': prob_arr
            })

        else:
            self.real_params.update({
                'reg_mono': np.zeros(n_features) if self.reg_mono is None else np.array(self.reg_mono)
            })

            if self.max_features > n_features:
                raise ValueError('max_features should be no more than n_features.')
            max_features = self.max_features
            if self.max_features == 0 and self.pred_prob == 0.0 and n_features < 16:
                max_features = np.max([np.floor(n_features/3), 1])
            self.real_params.update({
                'max_features': max_features
            })

            pred_prob = self.pred_prob
            if self.pred_prob == 0.0 and self.max_features == 0:
                pred_prob = 0.4
            mean_weight = 1.0 if pred_prob == 0.0 else pred_prob
            prob_arr = feature_weight * (n_features * mean_weight) / np.sum(feature_weight)
            self.real_params.update({
                'prob_arr': prob_arr
            })

        return self


    def _train_regression(self,
        X,
        y,
        sample_weight,
        feature_weight):
        """
        Returns
        -------
        self : object
            Returns self.
        """
        n_samples = self.n_samples_
        n_features = self.n_features_
        result = PyTrain.Regression(
            np.ascontiguousarray(X.transpose().reshape(X.size)),
            np.ascontiguousarray(np.reshape(y, n_samples)),
            n_samples,
            n_features,
            np.ascontiguousarray(self.real_params['presort_row']),
            np.ascontiguousarray(self.real_params['presort_rank']),
            np.ascontiguousarray(self.real_params['presort_inv_num']),
            self.n_estimators,
            self.real_params['n_to_sample'],
            np.ascontiguousarray(np.reshape(sample_weight, n_samples)),
            self.bootstrap,
            self.tree_block,
            self.min_samples_split,
            self.min_info_ratio,
            self.max_depth,
            self.real_params['max_features'],
            np.ascontiguousarray(np.reshape(self.real_params['prob_arr'], n_features)),
            np.ascontiguousarray(np.reshape(self.real_params['reg_mono'], self.real_params['reg_mono'].size))
        )
        self.estimators_ = result
        return self


    def _train_classification(self,
        X,
        y,
        sample_weight,
        feature_weight):
        """
        Returns
        -------
        self : object
            Returns self.
        """
        n_samples = self.n_samples_
        n_features = self.n_features_
        result = PyTrain.Classification(
            np.ascontiguousarray(X.transpose().reshape(X.size)),
            np.ascontiguousarray(np.reshape(y, n_samples)),
            n_samples,
            n_features,
            np.ascontiguousarray(self.real_params['presort_row']),
            np.ascontiguousarray(self.real_params['presort_rank']),
            np.ascontiguousarray(self.real_params['presort_inv_num']),
            self.n_estimators,
            self.real_params['n_to_sample'],
            np.ascontiguousarray(np.reshape(sample_weight, n_samples)),
            self.bootstrap,
            self.tree_block,
            self.min_samples_split,
            self.min_info_ratio,
            self.max_depth,
            self.real_params['max_features'],
            np.ascontiguousarray(np.reshape(self.real_params['prob_arr'], n_features)),
            np.ascontiguousarray(self.real_params['class_weight'])
        )
        self.estimators_ = result
        return self


    def predict(self, X):
        """Fit estimator.

        Parameters
        ----------
        X : array-like, shape=(n_samples, n_features)
            The input samples.

        Returns
        -------
        y_pred : array-like, shape=(n_samples)
            Returns the predicted result.
        """
        self.n_outputs_ = X.shape[1]
        if self.is_classifier:
            self.y_pred, self.y_pred_votes, self.y_pred_proba = self._predict_classification(X)
            return self.classes_[self.y_pred]
        else:
            self.y_pred = self._predict_regression(X)
            return self.y_pred


    def _predict_regression(self, X):
        result = PyPredict.Regression(np.ascontiguousarray(X.reshape(X.size)),
            X.shape[0],
            X.shape[1],
            self.estimators_['forest']['origin'],
            self.estimators_['forest']['facOrig'],
            self.estimators_['forest']['facSplit'],
            self.estimators_['forest']['forestNode'],
            self.estimators_['leaf']['yRanked'],
            self.estimators_['leaf']['leafOrigin'],
            self.estimators_['leaf']['leafNode'],
            self.estimators_['leaf']['bagRow'],
            self.estimators_['leaf']['nRow'],
            self.estimators_['leaf']['rank']
        )
        return result


    def _predict_classification(self, X):
        result = PyPredict.Classification(np.ascontiguousarray(X.reshape(X.size)),
            X.shape[0],
            X.shape[1],
            self.n_classes_,
            self.estimators_['forest']['origin'],
            self.estimators_['forest']['facOrig'],
            self.estimators_['forest']['facSplit'],
            self.estimators_['forest']['forestNode'],
            self.estimators_['leaf']['yLevels'],
            self.estimators_['leaf']['leafOrigin'],
            self.estimators_['leaf']['leafNode'],
            self.estimators_['leaf']['bagRow'],
            self.estimators_['leaf']['nRow'],
            self.estimators_['leaf']['weight']
        )
        return result


    def get_params(self, deep=True):
        """To make the estimator scikit-learn capable
        #TODO how to "don't repeat yourself"?
        """
        default_param_keys = self.__class__.get_allowed_param_keys()
        result = {}
        for k in default_param_keys:
            result[k] = getattr(self, k)
        return result


    def set_params(self, **kwargs):
        """To make the estimator scikit-learn capable"""
        default_param_keys = self.__class__.get_allowed_param_keys()
        for k, v in kwargs.items():
            if k in default_param_keys:
                setattr(self, k, v)
        return self



class PyboristClassifier(PyboristModel):
    """The classifier.

    Parameters
    ----------
    n_estimators: int, optional (default=10)
        The number of trees to train.

    bootstrap: bool, optional (default=True)
        Whether row sampling is by replacement.

    class_weight: str or None or array-like, optional (default=None)
        Proportional weighting of classification categories.
        Could use None or `balance`, or an array

    min_info_ratio: float, optional (default=0.01)
        Information ratio with parent below which node does not split.

    min_samples_split: int, optional (default=2)
        Minimum number of distinct row references to split a node.

    max_depth: int or None, optional (default=0)
        Maximum number of tree levels to train. Zero denotes no limit.

    no_validate: bool, optional (default=True)
        Whether to train without validation.

    n_to_sample: int, optional (default=0)
        Number of rows to sample, per tree.

    max_features: int, optional (default=0)
        Number of trial predictors for a split.

    pred_prob: float, optional (default=0.0)
        Probability of selecting individual predictor as trial splitter.
        Causes each predictor to be selected as a splitting candidate with distribution Bernoulli(pred_prob).

    quantiles_arr: array-like or None (default=None)
        Quantile levels to validate.

    q_bin: int, optional (default=5000)
        Bin size for facilating quantiles at large sample count.

    tree_block: int, optional (default=1)
        Maximum number of trees to train during a single level (e.g., coprocessor computing).

    pvt_block: int, optional (default=8)
        Maximum number of trees to train in a block (e.g., cluster computing).

    Attributes
    ----------
    n_features_ : int
        The number of features.

    n_samples_ : int
        The number of samples in trainning dataset.

    n_classes_ : int
        The number of classes in classification.

    classes_ : array-like
        The class labels extracted from the training response in classification.

    n_outputs_ : int
        The number of outputs.
    """

    _estimator_type = 'classifier'

    def __init__(self,
        n_estimators = 10,
        bootstrap = True,
        class_weight = None,
        min_info_ratio = 0.01,
        min_samples_split = 2,
        max_depth = 0,
        no_validate = True,
        n_to_sample = 0,
        max_features = 0,
        pred_prob = 0.0,
        quantiles_arr = None,
        q_bin = 5000,
        tree_block = 1,
        pvt_block = 8):
        super(PyboristClassifier, self).__init__(n_estimators = n_estimators,
            bootstrap = bootstrap,
            class_weight = class_weight,
            min_info_ratio = min_info_ratio,
            min_samples_split = min_samples_split,
            max_depth = max_depth,
            no_validate = no_validate,
            n_to_sample = n_to_sample,
            max_features = max_features,
            pred_prob = pred_prob,
            quantiles_arr = quantiles_arr,
            q_bin = q_bin,
            reg_mono = None,
            tree_block = tree_block,
            pvt_block = pvt_block,
            is_classifier = True)


    @staticmethod
    def get_allowed_param_keys():
        """
        Get the default params for this model.
        """
        keys = ['n_estimators',
            'bootstrap',
            'class_weight',
            'min_info_ratio',
            'min_samples_split',
            'max_depth',
            'no_validate',
            'n_to_sample',
            'max_features',
            'pred_prob',
            'quantiles_arr',
            'q_bin',
            'tree_block',
            'pvt_block']
        return keys


    def predict_proba(self, X, return_votes=False):
        """Fit estimator for classification.

        Parameters
        ----------
        X : array-like, shape=(n_samples, n_features)
            The input samples.

        return_votes: bool, optional (defualt=False)
            Whether return votes or probilities.

        Returns
        -------
        y_pred_proba or y_pred_votes: array-like, shape=(n_samples, n_classes)
            Returns the predicted result, in probilities or votes.
        """
        self.predict(X)
        if return_votes:
            return self.y_pred_votes
        return self.y_pred_proba



class PyboristRegressor(PyboristModel):
    """The classifier.

    Parameters
    ----------
    n_estimators: int, optional (default=10)
        The number of trees to train.

    bootstrap: bool, optional (default=True)
        Whether row sampling is by replacement.

    min_info_ratio: float, optional (default=0.01)
        Information ratio with parent below which node does not split.

    min_samples_split: int, optional (default=2)
        Minimum number of distinct row references to split a node.

    max_depth: int or None, optional (default=0)
        Maximum number of tree levels to train. Zero denotes no limit.

    no_validate: bool, optional (default=True)
        Whether to train without validation.

    n_to_sample: int, optional (default=0)
        Number of rows to sample, per tree.

    max_features: int, optional (default=0)
        Number of trial predictors for a split.

    pred_prob: float, optional (default=0.0)
        Probability of selecting individual predictor as trial splitter.
        Causes each predictor to be selected as a splitting candidate with distribution Bernoulli(pred_prob).

    quantiles_arr: array-like or None (default=None)
        Quantile levels to validate.

    q_bin: int, optional (default=5000)
        Bin size for facilating quantiles at large sample count.

    reg_mono: array-like or None, optional (default=None)
        Signed probability constraint for monotonic regression.

    tree_block: int, optional (default=1)
        Maximum number of trees to train during a single level (e.g., coprocessor computing).

    pvt_block: int, optional (default=8)
        Maximum number of trees to train in a block (e.g., cluster computing).

    Attributes
    ----------
    n_features_ : int
        The number of features.

    n_samples_ : int
        The number of samples in trainning dataset.

    n_outputs_ : int
        The number of outputs.
    """

    _estimator_type = 'regressor'

    def __init__(self,
        n_estimators = 10,
        bootstrap = True,
        min_info_ratio = 0.01,
        min_samples_split = 2,
        max_depth = 0,
        no_validate = True,
        n_to_sample = 0,
        max_features = 0,
        pred_prob = 0.0,
        quantiles_arr = None,
        q_bin = 5000,
        reg_mono = None,
        tree_block = 1,
        pvt_block = 8):
        super(PyboristRegressor, self).__init__(n_estimators = n_estimators,
            bootstrap = bootstrap,
            class_weight = None,
            min_info_ratio = min_info_ratio,
            min_samples_split = min_samples_split,
            max_depth = max_depth,
            no_validate = no_validate,
            n_to_sample = n_to_sample,
            max_features = max_features,
            pred_prob = pred_prob,
            quantiles_arr = quantiles_arr,
            q_bin = q_bin,
            reg_mono = reg_mono,
            tree_block = tree_block,
            pvt_block = pvt_block,
            is_classifier = False)


    @staticmethod
    def get_allowed_param_keys():
        """
        Get the default params for this model.
        """
        keys = ['n_estimators',
            'bootstrap',
            'min_info_ratio',
            'min_samples_split',
            'max_depth',
            'no_validate',
            'n_to_sample',
            'max_features',
            'pred_prob',
            'quantiles_arr',
            'q_bin',
            'reg_mono',
            'tree_block',
            'pvt_block']
        return keys
