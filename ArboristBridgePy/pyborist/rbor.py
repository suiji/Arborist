import numpy as np
from sklearn.utils import check_array

from .cyrowrank import PyRowRank


class PredTrainObj(object):
    """This is the normal Python object storing the predblock info.

    Parameters
    ----------
    x: array-like
        The 2-D design matrix.

    Attributes
    ----------
    colNames: array-like
        The array of the names of cols.

    rowNames: array-like
        The array of the names of rows.

    blockNum: array-like
        The numpy array/matrix of the blocks containing numeric predictors.

    nPredNum: int (default = 0)
        The number of the numeric predictors.

    blockFac: array-like
        The numpy array/matrix of the blocks containing factoric predictors.

    nPredFac: int (default = 0)
        The number of the factoric predictors.

    nRow: int (defualt = 0)
        Number of rows
    """
    def __init__(self, x):
        x = check_array(x).astype(np.float)
        self.colNames = None
        self.rowNames = None
        self.blockNum = x
        self.nPredNum = x.shape[1]
        self.blockFac = None
        self.nPredFac = 0
        self.nRow = x.shape[0]
        self._initRowRank()


    def _initRowRank(self):
        """Call the backend to generate the rowrank. Similar to the R verson.

        Attributes
        ----------
        row: array-like

        rank: array-like

        invNum: array-like
        """
        nRow = self.nRow
        nPredNum = self.nPredNum
        nPredFac = self.nPredFac
        nPred = nPredNum + nPredFac
        rank = np.empty([nPred * nRow], dtype=np.int)
        row = np.empty([nPred * nRow], dtype=np.int)
        invNum = np.empty([nPred * nRow], dtype=np.int)
        if nPredNum > 0:
            PyRowRank.PreSortNum(np.reshape(self.blockNum.transpose(),
                    (nPred * nRow), 'C'),
                nPredNum,
                nRow,
                row,
                rank,
                invNum)
        if nPredFac > 0:
            PyRowRank.PreSortFac(np.reshape(self.blockFac.transpose(),
                    (nPred * nRow), 'C'),
                nPredNum,
                nPredFac,
                nRow,
                row,
                rank)
        self.row = row
        self.rank = rank
        self.invNum = invNum


