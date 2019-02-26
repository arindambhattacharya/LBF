from sklearn.linear_model import SGDClassifier
import bloom_filter as bf
import numpy as np
import copy
import sys
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
ch = logging.StreamHandler()
ch.setLevel(logging.WARNING)

formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


class BloomClassifier(object):
    """
    Bloom Classifier
    @param clf: A pretrained initial classifier. Optional.
    @param method: 0: Baseline.
                   1: CA-LBF I - Incremental retraining
                   2: CA-LBF II - Separate classifier
    """

    def __init__(self, clf=None, method=1, retrain_threshold=1000):
        super(BloomClassifier, self).__init__()
        if clf:
            self.CUSTOMCLF = True
            self.clfs = [clf[1]]
            self.clf_name = clf[0]
        else:
            self.CUSTOMCLF = False
            logging.info('Using default classifier...')
            self.clfs = [SGDClassifier(loss='log', max_iter=5, tol=1e-3)]
        self.to_be_inserted = []
        self.count = 0
        self.method = method

        if self.method > 0:
            self.retrain_threshold = retrain_threshold
        else:
            self.retrain_threshold = 1

    def initialize(self, X_train, Y_train, n=0, m=1000, k=3):
        if not self.CUSTOMCLF:
            logging.info('Training default classifier')
            self.clfs[0].fit(X_train, Y_train)
        false_neg = []
        Y_pred = self.clfs[0].predict(X_train)
        # false_neg_bits = [
        #     1 if y and not y_pred else 0 for y_pred, y in zip(Y_pred, Y_train)
        # ]
        for i in range(len(X_train)):
            if Y_train[i] == '1':
                if Y_pred[i] == '0':
                    false_neg.append(X_train[i])
        # false_neg = [x for x, fn in zip(X_train, false_neg_bits) if fn]
        p = 0.01
        if not n:
            n = len(false_neg)
            self.overflow_filter = bf.BloomFilter(n=n, p=p, m=m, k=k)

        logger.debug('Setting Bloom filter capacity to: {}'.format(n))
        logger.debug('Setting Bloom filter size to: {}'.format(m))
        logger.debug('Setting Bloom filter fpr to: {}'.format(p))

        for x in false_neg:
            self.overflow_filter.insert(x)

    def insert(self, x):
        self.to_be_inserted.append(x)
        self.count += 1
        if self.count % self.retrain_threshold == 0:
            self._retrain_and_insert()
            self.to_be_inserted = []

    def _retrain_and_insert(self):
        if self.method == 1:
            logging.info('Retraining...')
            # y = np.ones(len(self.to_be_inserted)).as_type(np.str)
            y = ['1'] * len(self.to_be_inserted)
            y[0] = '0'
            y = np.array(y)
            clf = copy.deepcopy(self.clfs[-1])
            if self.method == 1:
                clf.fit(self.to_be_inserted, y)
            logging.info('Retraining done!')

            for x in self.to_be_inserted:
                for clf in self.clfs:
                    if clf.predict([x]):
                        break
                else:
                    self.overflow_filter.insert(x)
        elif self.method == 2:
            logging.info('New small classifier')
        else:
            for x in self.to_be_inserted:
                self.overflow_filter.insert(x)

    def check(self, x):
        for clf in self.clfs:
            if clf.predict([x]) == '1':
                return True
        return self.overflow_filter.check(x)

    def get_fpr(self, X, Y):
        fp = len([x for x, y in zip(X, Y) if self.check(x) and y == '0'])
        n = len([y for y in Y if y == '0'])
        if n:
            return fp / n
        return 0

    def get_size(self):
        return sys.getsizeof(self.clfs) + sys.getsizeof(self.overflow_filter)
