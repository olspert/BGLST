import unittest
import numpy as np
from BGLST import BGLST
import scipy.linalg as la
from scipy import stats

class TestBGLST(unittest.TestCase):

    ''' 
    Tests whether the posterior means and variances of the parameters are correct
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
        # for simplicity of testing use constant variance
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
        (tau, mean, cov, y_model, loglik, _) = bglst.model(freq)

        # obtain the posterior means and variances using linear algebra
        w0 = np.array([A_hat, B_hat, alpha_hat, beta_hat])
        V0 = np.diag([1.0/w_A, 1.0/w_B, 1.0/w_alpha, 1.0/w_beta])
        X = np.column_stack((np.cos(t*2.0*np.pi*freq - tau), 
                             np.sin(t*2.0*np.pi*freq - tau), 
                            t, 
                            np.ones(n)))
                            
                            
        V0inv = la.inv(V0)
        sigma2 = sigma*sigma
        Vn = la.inv(V0inv*sigma2 + np.dot(X.T,X))*sigma2
        wn = np.dot(np.dot(Vn, V0inv), w0) + np.dot(np.dot(Vn, X.T), y)/sigma2

        # check if the results match
        np.testing.assert_allclose(mean, wn, rtol=1e-10, atol=0)
        np.testing.assert_allclose(cov, Vn, rtol=1e-10, atol=0)


    ''' 
    Tests the correctness of the posterior predictive
    '''
    def test_posterior_predictive(self):
        # generate some data
        time_range = 1.0#200.0
        n = 300
        t = np.random.uniform(0.0, time_range, n)
        t = np.sort(t)
        duration = max(t) - min(t)
        freq = 1.0/np.random.uniform(0.0, time_range/5)
        true_slope = 1.0/np.random.uniform(100, 200)
        offset = np.random.uniform(-1, 1)
        # for simplicity of testing use constant variance
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
        n_test = 10
        t_test = np.linspace(0, time_range, n_test)
        sigma_test = sigma
        w_test = np.ones(n_test)/sigma_test**2
        (tau, _, _, pred_mean, _, pred_var) = bglst.model(freq, t=t_test, w=w_test, calc_pred_var=True)

        # obtain the posterior means and variances using linear algebra
        w0 = np.array([A_hat, B_hat, alpha_hat, beta_hat])
        V0 = np.diag([1.0/w_A, 1.0/w_B, 1.0/w_alpha, 1.0/w_beta])
        X = np.column_stack((np.cos(t*2.0*np.pi*freq - tau), 
                             np.sin(t*2.0*np.pi*freq - tau), 
                            t, 
                            np.ones(n)))
                            
                            
        V0inv = la.inv(V0)
        sigma2 = sigma**2
        Xt = np.transpose(X)
        Vn = la.inv(V0inv*sigma2 + np.dot(Xt,X))*sigma2
        wn = np.dot(np.dot(Vn, V0inv), w0) + np.dot(np.dot(Vn, Xt), y)/sigma2

        X_test = np.column_stack((np.cos(t_test*2.0*np.pi*freq - tau), 
                             np.sin(t_test*2.0*np.pi*freq - tau), 
                            t_test, 
                            np.ones(n_test)))

        sigma_test2 = sigma_test**2
        pred_mean_expected = np.dot(wn, X_test.T)
        pred_var_expected = sigma_test2 + np.einsum("ij,ij->j",X_test.T, np.dot(Vn, X_test.T))
        
        # check if the results match
        np.testing.assert_allclose(pred_mean, pred_mean_expected, rtol=1e-10, atol=0)
        np.testing.assert_allclose(pred_var, pred_var_expected, rtol=1e-10, atol=0)
        

if __name__ == '__main__':
    unittest.main()