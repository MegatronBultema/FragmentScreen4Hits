import numpy as np


class NMF(object):
    '''
    A Non-Negative Matrix Factorization (NMF) model using the Alternating Least
    Squares (ALS) algorithm.
    This class represents an NMF model, which is a useful unsupervised data
    mining tool; e.g. for finding latent topics in a text corpus such as NYT
    articles.
    '''

    def __init__(self, k, max_iters=50, alpha=0.5, eps=1e-6):
        '''
        Constructs an NMF object which will mine for `k` latent topics.
        The `max_iters` parameter gives the maximum number of ALS iterations
        to perform. The `alpha` parameter is the learning rate, it should range
        in (0.0, 1.0]. `alpha` near 0.0 causes the model parameters to be
        learned slowly over many many ALS iterations, while an alpha near 1.0
        causes model parameters to be fit quickly over very few ALS iterations.
        '''
        self.k = k
        self.max_iters = max_iters
        self.alpha = alpha
        self.eps = eps

    def _fit_one(self, V):
        '''
        Do one ALS iteration. This method updates self.H and self.W
        and returns None.
        '''
        # Fit H while holding W constant:
        H_new = np.linalg.lstsq(self.W, V)[0].clip(min=self.eps)
        self.H = self.H * (1.0 - self.alpha) + H_new * self.alpha

        # Fit W while holding H constant:
        W_new = np.linalg.lstsq(self.H.T, V.T)[0].T.clip(min=self.eps)
        self.W = self.W * (1.0 - self.alpha) + W_new * self.alpha

    def fit(self, V, verbose = False):
        '''
        Do many ALS iterations to factorize `V` into matrices `W` and `H`.
        Let `V` be a matrix (`n` x `m`) where each row is an observation
        and each column is a feature. `V` will be factorized into a the matrix
        `W` (`n` x `k`) and the matrix `H` (`k` x `m`) such that `WH` approximates
        `V`.
        This method returns the tuple (W, H); `W` and `H` are each ndarrays.
        '''
        n, m = V.shape
        self.W = np.random.uniform(low=0.0, high=1.0 / self.k, size=(n, self.k))
        self.H = np.random.uniform(low=0.0, high=1.0 / self.k, size=(self.k, m))
        for i in range(self.max_iters):
            if verbose:
                print 'iter', i, ': reconstruction error:', self.reconstruction_error(V)
            self._fit_one(V)
        if verbose:
            print 'FINAL reconstruction error:', self.reconstruction_error(V), '\n'
        return self.W, self.H

    def reconstruction_error(self, V):
        '''
        Compute and return the reconstruction error of `V` as the
        matrix L2-norm of the residual matrix.
        See: https://en.wikipedia.org/wiki/Matrix_norm
        See: https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.norm.html
        '''
        return np.linalg.norm(V - self.W.dot(self.H))

if __name__ == '__main__':
    '''
    Record of what I ran:
    import process_data as proc
    data = proc.read_data()
    f1, yfill = proc.f1_yfill(data)
     X_train, X_test, y_train, y_test =train_test_split(f1, yfill, test_size=0.20, random_state=42, stratify =yfill)
     factorizer = nmf.NMF(k=10, max_iters=500, alpha=0.1)
     W, H = factorizer.fit(X_train.values, verbose=True)
     clusters are H.... how to I visualize this? or maybe I sould be using the W matix as new features for fitting NN or RF

     f1 vs yfill -
        FINAL reconstruction error: 475.109703439
     all features vs yfill -
        factorizer = nmf.NMF(k=10, max_iters=1000, alpha=0.1)
        FINAL reconstruction error: 463.098091213
     bits, yfill -
        FINAL reconstruction error: 88.7998808882 (!)
        heatmap_W and heatmap_H saved, ran SOM on W but no clustering

    I think maybe 10 is to many for k will now try 3
    bits yfill
        factorizer = nmf.NMF(k=3, max_iters=500, alpha=0.1)
        FFINAL reconstruction error: 94.4946592695

    trained 3 with f1 features
    FINAL reconstruction error: 71.6197318232
    tried to run W with 3 features through the SOM but no good seperation visible
      whould check with Adam if I should try a different SOM construction (maybe less space )

    **or**
    use a 10 feature NMF for input into RF or NN or LR model_selection

    **or**
    run a SVM for features to input into models/SOM

    **or**
    try clustering of new features?

    '''
