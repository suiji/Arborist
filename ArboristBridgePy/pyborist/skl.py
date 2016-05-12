import numpy as np
from sklearn.utils import check_array, assert_all_finite, check_X_y

from .cyrowrank import PyRowRank

__all__ = ['PyboristModel']



class PyboristModel(object):
    """The scikit-learn API for Pyborist model.

    Parameters
    ----------
    n_estimators: int, optional (default=10)
        The number of trees to train.

    bootstrap: bool, optional (defualt=True)
        Thether row sampling is by replacement.

    categorical_census: str, optional (default='votes')
        Report categorical validation by votes or by probability.
        Could use `prob` or `votes`.

    class_weight: str or None or array-like, optional (default=None)
        Proportional weighting of classification categories.
        Could use None or `balance`, or an array

    min_info_ratio: float, optional (default=0.01)
        Information ratio with parent below which node does not split.

    min_samples_split: int, optional (default=2 for regression or 5 for classification)
        Minimum number of distinct row references to split a node.

    max_depth: int or None, optional (default=0)
        Maximum number of tree levels to train. Zero denotes no limit.

    no_validate: bool, optional (default=True)
        Whether to train without validation.

    n_to_sample: int, optional (default=0)
        Number of rows to sample, per tree.

    pred_fixed: int, optional (default=0)
        Number of trial predictors for a split (`mtry`).

    pred_prob: float, optional (default=0.0)
        Probability of selecting individual predictor as trial splitter.
        Causes each predictor to be selected as a splitting candidate with distribution Bernoulli(pred_prob).

    quantiles_arr: array-like or None (dafault=None)
        Quantile levels to validate.

    report_quantiles: bool, optonal (default = False)
        Whether to report quantiles at validation.

    q_bin: int, optional (default=5000)
        Bin size for facilating quantiles at large sample count.

    reg_mono: array-like or None, optional (default=None)
        Signed probability constraint for monotonic regression.

    tree_block: int, optional (default=1)
        Maximum number of trees to train during a single level (e.g., coprocessor computing).

    pvt_block: int, optional (default=8)
        Maximum number of trees to train in a block (e.g., cluster computing).

    is_classify_task: bool, optional (default = False)
        Regression or Classification?

    Attributes
    ----------
    n_features_ : int
        The number of features.

    n_outputs_ : int
        The number of outputs.

    """
    def __init__(self,
        n_estimators = 10,
        bootstrap = True,
        categorical_census = 'votes',
        class_weight = None,
        min_info_ratio = 0.01,
        min_samples_split = 0,
        max_depth = 0,
        no_validate = True,
        n_to_sample = 0,
        pred_fixed = 0,
        pred_prob = 0.0,
        quantiles_arr = None,
        report_quantiles = False,
        q_bin = 5000,
        reg_mono = None,
        tree_block = 1,
        pvt_block = 8,
        is_classify_task = False):
        # a trick to save everything into self...
        for k, v in locals().items():
            if k == 'self':
                continue
            setattr(self, k, v)

        if self.min_samples_split == 0:
            if self.is_classify_task:
                self.min_samples_split = 2
            else:
                self.min_samples_split = 5

        if self.min_samples_split <= 0:
            raise ValueError('Invalid min_samples_split.')

        if self.is_classify_task and self.quantiles_arr is not None:
            raise ValueError('Quantiles are for regression only.')
        if self.quantiles_arr is not None:
            if np.any(self.quantiles_arr < 0.0) or np.any(self.quantiles_arr > 1.0):
                raise ValueError('Quantiles shoule be inside 0 and 1.')
            if np.any(np.diff(self.quantiles_arr) < 0.0):
                raise ValueError('Quantiles should be increasing.')

        if self.pred_prob < 0.0 or self.pred_prob > 1.0:
            raise ValueError('pred_prob should be inside [0.0, 1.0].')


    def fit(self, X, y, sample_weight = None, feature_weight = None):
        """Fit estimator.

        Parameters
        ----------
        X : array-like, shape=(n_samples, n_features)
            The input samples.

        y: array-like, shape=(n_samples)

        sample_weight: array-like, shape=(n_samples) or None

        feature_weight: array-like, shape=(n_features) or None
            Relative weighting of individual predictors as trial splitters.

        Returns
        -------
        self : object
            Returns self.
        """
        X, y = check_X_y(X, y)
        X = X.astype(np.float)
        self.sample_weight = sample_weight
        self.feature_weight = feature_weight

        self._init_basical_attrbutes(X)
        self._init_row_rank()
        self._adjust_model_params()

        if self.is_classify_task:
            if np.any(self.reg_mono != 0.0):
                raise ValueError('Categorical response could not have reg_mono.')
            self._train_classification()
        else:
            self._train_regression()

        return self

    def _init_basical_attrbutes(self, X):
        """
        Parameters
        ----------
        X: array-like
            The 2-D design matrix.

        Attributes
        ----------
        col_names: array-like
            The array of the names of cols.

        row_names: array-like
            The array of the names of rows.

        block_num: array-like
            The numpy array/matrix of the blocks containing numeric predictors.

        n_features_num: int (default = 0)
            The number of the numeric predictors.

        block_fac: array-like
            The numpy array/matrix of the blocks containing factoric predictors.

        n_features_fac: int (default = 0)
            The number of the factoric predictors.

        n_samples: int (defualt = 0)
            Number of rows

        n_features: int (default = 0)
            The number of predictors.
        """
        self.col_names = None
        self.row_names = None
        self.block_num = X
        self.n_features_num = X.shape[1]
        self.block_fac = None
        self.n_features_fac = 0
        self.n_samples = X.shape[0]
        self.n_features = self.n_features_num + self.n_features_fac

    def _init_row_rank(self):
        """Call the backend to generate the rowrank. Similar to the R verson.

        Attributes
        ----------
        row: array-like

        rank: array-like

        inv_num: array-like
        """
        n_samples = self.n_samples
        n_features_num = self.n_features_num
        n_features_fac = self.n_features_fac
        n_features = self.n_features
        rank = np.empty([n_features * n_samples], dtype=np.int)
        row = np.empty([n_features * n_samples], dtype=np.int)
        inv_num = np.empty([n_features * n_samples], dtype=np.int)
        if n_features_num > 0:
            PyRowRank.PreSortNum(np.reshape(self.block_num.transpose(),
                    (n_features * n_samples), 'C'),
                n_features_num,
                n_samples,
                row,
                rank,
                inv_num)
        if n_features_fac > 0:
            PyRowRank.PreSortFac(np.reshape(self.block_fac.transpose(),
                    (n_features * n_samples), 'C'),
                n_features_num,
                n_features_fac,
                n_samples,
                row,
                rank)
        self.row = row
        self.rank = rank
        self.inv_num = inv_num

    def _adjust_model_params(self):
        """Regenerate some parameters of model based on input.
        """
        #TODO params should be invariant

        if self.min_samples_split > self.n_samples:
            raise ValueError('Invalid min_samples_split.')

        if self.reg_mono is None:
            self.reg_mono = np.zeros(self.n_features)

        if self.n_to_sample == 0:
            if self.bootstrap:
                self.n_to_sample = self.n_samples
            else:
                self.n_to_sample = np.round((1-np.exp(-1)) * self.n_samples)

        if self.pred_fixed == 0 \
            and self.pred_prob == 0 and self.n_features < 16:
            if not self.is_classify_task:
                self.pred_fixed = np.max([np.floor(self.n_features/3), 1])
            else:
                self.pred_fixed = np.floor(np.sqrt(self.n_features))

        if self.pred_prob == 0.0 and self.pred_fixed == 0:
            if not self.is_classify_task:
                self.pred_prob = 0.4
            else:
                self.pred_prob = np.ceil(np.sqrt(self.n_features)) / self.n_features

        if self.pred_fixed > self.n_features:
            raise ValueError('pred_fixed should be no more than n_features.')

        if self.is_classify_task:
            ctg_width = np.max(y) - 1 # how many categoriess
            if self.class_weight is None:
                self.class_weight = np.ones(ctg_width)
            elif self.class_weight == 'balanced':
                self.class_weight = np.zeros(ctg_width)
            elif len(self.class_weight) != ctg_width:
                raise ValueError('Invalid class weighting.')
            elif np.any(self.class_weight < 0.0):
                raise ValueError('Invalid class weighting.')
            elif np.sum(self.class_weight) == 0.0:
                raise ValueError('Invalid class weighting.')
            else:
                raise ValueError('Invalid class weighting.')
        else:
            #TODO maybe move some code into __init__
            if self.class_weight is not None:
                raise ValueError('Class Weight should be used in classification.')

        if self.feature_weight is None:
            self.feature_weight = np.ones(self.n_features)
        else:
            if len(self.feature_weight) != self.n_features:
                raise ValueError('Predictor weight should equal to column number.')
            if np.any(self.feature_weight < 0.0):
                raise ValueError('Predictor weight should larger than zero.')
            if np.sum(self.feature_weight) == 0.0:
                raise ValueError('Predictor weight could not be all zero.')

        if self.sample_weight is None:
            self.sample_weight = np.ones(self.n_samples)
        else:
            if len(self.sample_weight) != self.n_samples:
                raise ValueError('Sample weight should equal to column number.')
            if np.any(self.sample_weight < 0.0):
                raise ValueError('Sample weight should larger than zero.')
            if np.sum(self.sample_weight) == 0.0:
                raise ValueError('Sample weight could not be all zero.')

        mean_weight = 1.0 if self.pred_prob == 0.0 else self.pred_prob
        self.prob_arr = self.feature_weight * (self.n_features * mean_weight) / np.sum(self.feature_weight)


    def _train_classification(self):
        pass

    def _train_regression(self,
        block_num,
        n_features_num,
        block_fac,
        n_features_fac,
        n_features,

        row,
        rank,
        inv_num,

        y,

        n_estimators,
        n_to_sample,
        sample_weight,
        bootstrap,
        tree_block,
        min_samples_split,
        min_info_ratio,
        max_depth,
        pred_fixed,
        prob_arr,
        reg_mono
        ):
        pass

    def predict(self, X):
        """Fit estimator.

        Parameters
        ----------
        X : array-like, shape=(n_samples, n_features)
            The input samples.

        Returns
        -------
        self : object
            Returns self.
        """
        return self
