import unittest
import numpy as np
from BGLST import BGLST
import scipy.linalg as la
from scipy import stats

class TestBGLST(unittest.TestCase):

    ''' 
    Tests that the posterior means and variances of the parameters are correct
    '''
    def test_posterior(self):
        # generate some data
        time_range = 1.0#200.0
        n = 300
        t = np.random.uniform(0.0, time_range, n)
        t = np.sort(t)
        duration = max(t) - min(t)
        freq = 1.0/np.random.uniform(0.0, time_range/5)
        true_slope = 1.0/np.random.uniform(100, 200)
        offset = np.random.uniform(-1, 1)
        sigma = 0.1
        epsilon = np.random.normal(0, sigma, n)
        y = np.cos(2*np.pi*freq*t) + true_slope*t + offset + epsilon
        w = np.ones(n)/sigma**2
        
        # set the prior means and variances of the parameters
        slope, intercept, r_value, p_value, std_err = stats.linregress(t, y)
        w_A = 2.0/np.var(y)
        A_hat = 0.0
        w_B = 2.0/np.var(y)
        B_hat = 0.0
        w_alpha = duration**2/np.var(y)
        alpha_hat = slope
        w_beta = 1.0/(np.var(y) + intercept**2)
        beta_hat = intercept

        # initialie model
        bglst = BGLST(t, y, w, 
                            w_A = w_A, A_hat = A_hat,
                            w_B = w_B, B_hat = B_hat,
                            w_alpha = w_alpha, alpha_hat = alpha_hat, 
                            w_beta = w_beta, beta_hat = beta_hat)

        # fit the model using some frequency (in this case the true frequency)
        # and obtain the posterior means and variances of the parameters
        (tau, mean, cov, y_model, loglik) = bglst.model(freq)

        # obtain the posterior means and variances using linear algebra
        w0 = np.array([A_hat, B_hat, alpha_hat, beta_hat])
        V0 = np.diag([1.0/w_A, 1.0/w_B, 1.0/w_alpha, 1.0/w_beta])
        X = np.column_stack((np.cos(t*2.0*np.pi*freq - tau), 
                             np.sin(t*2.0*np.pi*freq - tau), 
                            t, 
                            np.ones(n)))
                            
                            
        V0inv = la.inv(V0)
        sigma2 = sigma*sigma
        Xt = np.transpose(X)
        Vn = la.inv(V0inv*sigma2 + np.dot(Xt,X))*sigma2
        wn = np.dot(np.dot(Vn, V0inv), w0) + np.dot(np.dot(Vn, Xt), y)/sigma2

        # check if the results match
        np.testing.assert_allclose(mean, wn, rtol=1e-10, atol=0)
        np.testing.assert_allclose(cov, Vn, rtol=1e-10, atol=0)
        

if __name__ == '__main__':
    unittest.main()