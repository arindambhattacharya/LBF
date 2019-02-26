from bitarray import bitarray
import mmh3
import math


class BloomFilter():
    """Implementation of bloom filter

    """

    def __init__(self, m=0, n=0, k=3, p=0, clf=None):
        """Initialize bloom filter

        :param m: size of the bloom filter
        :param n: expected number of elements
        :param k: number of hash functions
        :param p: desired false positive rate

        """
        super(BloomFilter, self).__init__()
        if not m:
            self.size = self.get_size(n, p)
        else:
            self.size = m
        self.bloom_filter = bitarray(self.size)
        if not k:
            self.hash_count = self.get_hash_count(self.size, n)
        else:
            self.hash_count = k
        self.n = n
        if not p:
            self.fpr = self.get_fpr(n, m, k)
        else:
            self.fpr = p
        self.bloom_filter.setall(False)
        self.clf = clf

    def __str__(self):
        return 'Bloom filter initialized with: \nm: {}\nn: {}\nk: {}\np: {}\nClassifier: {}'.format(
            self.size, self.n, self.hash_count, self.fpr, self.clf)

    def __repr__(self):
        return self.__str__()

    def get_hash_count(self, m, n):
        '''
        Return the number of hash functions to be used using
        following formula
        k = (m/n) * lg(2)

        m : int
            size of bit array
        n : int
            number of items expected to be stored in filter
        '''
        k = (m / n) * math.log(2)
        return int(k)

    def get_size(self, n, p):
        '''
        Return the size of bit array(m) to used using
        following formula
        m = -(n * lg(p)) / (lg(2)^2)
        n : int
            number of items expected to be stored in filter
        p : float
            False Positive probability in decimal
        '''
        m = -(n * math.log(p)) / (math.log(2)**2)
        return int(m)

    def get_clf_hash(self, item):
        pass

    def get_fpr(self, n, m, k):
        return (1 - (1 - 1 / m)**(k * n))**k

    def insert(self, item, clf_hash=False):
        """Insert and element in the bloom filter

        :param item: item to be inserted
        :returns: Nothing
        :rtype: None
        """
        bits = []
        if not clf_hash:
            for i in range(self.hash_count):
                bits.append(mmh3.hash(item.tostring(), i) % self.size)
        else:
            bits.append(self.get_clf_hash(item))
        for b in bits:
            self.bloom_filter[b] = True

    def check(self, item):
        '''
        Check for existence of an item in filter
        '''
        for i in range(self.hash_count):
            digest = mmh3.hash(item.tostring(), i) % self.size
            if not self.bloom_filter[digest]:
                return False
        return True
