class PyboristModel(object):
    """The scikit-learn API for Pyborist model.

    Parameters
    ----------
    n_estimators: int, optional (default=10)
        The number of trees to train.

    bootstrap: bool, optional (defualt=True)
        Thether row sampling is by replacement.

    categorical_census: str, optional (default='prob')
        Report categorical validation by votes or by probability.
        Could use `prob` or `votes`.

    class_weight: str or None, optional (default=None)
        Proportional weighting of classification categories.
        Could use None or `balance`.

    min_info_ratio: float, optional (default=0.01)
        Information ratio with parent below which node does not split.

    min_samples_split: int, optional (default=2)
        Minimum number of distinct row references to split a node.

    max_depth: int or None, optional (default=0)
        Maximum number of tree levels to train. Zero denotes no limit.

    no_validate: bool, optional (default=True)
        Whether to train without validation.

    n_sample: int, optional (default=0)
        Number of rows to sample, per tree.

    pred_fixed: int, optional (default=0)
        Number of trial predictors for a split (`mtry`).

    pred_prob: float, optional (default=0.0)
        Probability of selecting individual predictor as trial splitter.
        Causes each predictor to be selected as a splitting candidate with distribution Bernoulli(pred_prob).

    pred_weight: array-like or None, optional (default=None)
        Relative weighting of individual predictors as trial splitters.

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
        categorical_census = 'prob',
        class_weight = None,
        min_info_ratio = 0.01,
        min_samples_split = 2,
        max_depth = 0,
        no_validate = True,
        n_sample = 0,
        pred_fixed = 0,
        pred_prob = 0.0,
        pred_weight = None,
        quantiles_arr = None,
        report_quantiles = False,
        q_bin = 5000,
        reg_mono = None,
        tree_block = 1,
        pvt_block = 8):
        pass

    def fit(self, X, y, sample_weight = None):
        """Fit estimator.

        Parameters
        ----------
        X : array-like, shape=(n_samples, n_features)
            The input samples.

        y: array-like, shape=(n_samples)

        sample_weight: array-like, shape=(n_samples) or None

        Returns
        -------
        self : object
            Returns self.
        """
        return self


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
