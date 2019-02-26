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
    @param model: A pretrained initial classifier.
    """

    def __init__(self, model):
        super(BloomClassifier, self).__init__()
        self.model = model
        self.to_be_inserted = []
        self.count = 0

    def initialize(self, x, y, n=0, m=1000, k=3):
        self.overflow_filter = bf.BloomFilter(m, n, k, p=0.01)
        for i in range(len(x)):
            if np.allclose(y[i], np.array([1, 0])): #y = [p_true, p_false]
                self.insert_one(x[i])

    def insert(self, x):
        # self.to_be_inserted.append(x)
        # self.count += 1
        # if self.count % self.retrain_threshold == 0:
        #     self._retrain_and_insert()
        #     self.to_be_inserted = []
        self.insert_one(x)

    def insert_one(self, idx):
        y1 = self.model.predict([idx]) #predict returns [p_true, p_false]
        try:
            y2 = self.model2.predict([idx])
        except Exception() as e:
            pass

        if np.argmax(y1) == 1:
            if y2:
                if np.argmax(y2) == 1:
                    self.overflow_filter.insert(idx)
            else:
                self.overflow_filter.insert(idx)

    def add_data(self, x, y, model):
        """Add new classifier along with new data

        :param x: x
        :param y: y
        :param model: new model. based on method could be incremental or from scratch
        """
        self.model2 = model
        for i in range(len(x)):
            if np.allclose(y[i], np.array([1, 0])): #y = [p_true, p_false]
                self.insert_one(x[i])


    # def _retrain_and_insert(self):
    #     if self.method == 1:
    #         logging.info('Retraining...')
    #         # y = np.ones(len(self.to_be_inserted)).as_type(np.str)
    #         y = ['1'] * len(self.to_be_inserted)
    #         y[0] = '0'
    #         y = np.array(y)
    #         clf = copy.deepcopy(self.clfs[-1])
    #         if self.method == 1:
    #             clf.fit(self.to_be_inserted, y)
    #         logging.info('Retraining done!')

    #         for x in self.to_be_inserted:
    #             for clf in self.clfs:
    #                 if clf.predict([x]):
    #                     break
    #             else:
    #                 self.overflow_filter.insert(x)
    #     elif self.method == 2:
    #         logging.info('New small classifier')
    #     else:
    #         for x in self.to_be_inserted:
    #             self.overflow_filter.insert(x)

    def check(self, x):
        if np.argmax(self.model.predict([x])) == '0':
                return True
        if np.argmax(self.model2.predict([x])) == '0':
                return True
        return self.overflow_filter.check(x)

    def get_fpr(self, X, Y):
        fp = len([x for x, y in zip(X, Y) if self.check(x) and np.allclose(y, np.array([0, 1]))])
        n = len([y for y in Y if np.allclose(y, np.array([0, 1]))])
        return fp / n

    def get_size(self):
        return sys.getsizeof(self.clfs) + sys.getsizeof(self.overflow_filter)
