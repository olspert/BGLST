import numpy as np
import matplotlib.pyplot as plt
import BGLST
from scipy import stats

axis_label_fs = 15
panel_label_fs = 15


time_range = 200.0
n = 300
t = np.random.uniform(0.0, time_range, n)
t = np.sort(t)
duration = max(t) - min(t)
true_freq = 1.0/np.random.uniform(0.001, time_range/5)
true_slope = 1.0/np.random.uniform(100, 200)
true_offset = np.random.uniform(-1, 1)
sigma = 0.1
epsilon = np.random.normal(0, sigma, n)
y = np.cos(2 * np.pi * true_freq * t) + true_slope*t + true_offset + epsilon
w = np.ones(n)/sigma**2

freq_start = 0.001
freq_end = 2.0 * true_freq
freq_count = 1000

slope, intercept, r_value, p_value, std_err = stats.linregress(t, y)
bglst = BGLST.BGLST(t, y, w, 
                    w_A = 2.0/np.var(y), A_hat = 0.0,
                    w_B = 2.0/np.var(y), B_hat = 0.0,
                    w_alpha = duration**2 / np.var(y), alpha_hat = slope, 
                    w_beta = 1.0 / (np.var(y) + intercept**2), beta_hat = intercept)

(freqs, probs) = bglst.calc_all(freq_start, freq_end, freq_count)

max_prob = max(probs)
max_prob_index = np.argmax(probs)
opt_freq = freqs[max_prob_index]

fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=False)

ax1.scatter(t, y, marker='+', color ='k', lw=0.5)
t_model = np.linspace(min(t), max(t), 1000)
w_model = np.ones(1000)/sigma**2
tau, mean, cov, y_model, loglik, pred_var = bglst.model(freq=opt_freq, t=t_model, w=w_model, calc_pred_var=True)
A = mean[0]
B = mean[1]
alpha = mean[2]
beta = mean[3]
bic = 2 * loglik - np.log(n) * 5
t_model = np.linspace(min(t), max(t), 1000)
std2 = np.sqrt(pred_var)*2.0
# plot the mean curve of the model
ax1.plot(t_model, y_model, 'r-')
# plot 90% confidence interval
ax1.fill_between(t_model, y_model-std2, y_model+std2, alpha=0.2, facecolor='lightsalmon', interpolate=True)

# plot the linear trend of the model
ax1.plot(t_model, alpha*t_model + beta, 'r--')
# plot the true trend
ax1.plot(t_model, true_slope*t_model + true_offset, 'k:')

min_prob = min(probs)
norm_probs = (probs - min_prob) / (max_prob - min_prob)
# plot the spectrum
ax2.plot(freqs, norm_probs, 'r-')
# plot the location of estimated frequency
ax2.plot([opt_freq, opt_freq], [0, norm_probs[max_prob_index]], 'r--')
# plot the location of true frequency
ax2.plot([true_freq, true_freq], [0, norm_probs[max_prob_index]], 'k:')

# print out the true and estimated parameter values
print "Frequency:", true_freq, opt_freq
print "Slope:", true_slope, alpha
print "Offset:", true_offset, beta

ax1.set_xlabel(r'$t$', fontsize=axis_label_fs)
ax1.set_ylabel(r'$y$', fontsize=axis_label_fs)
ax1.set_xlim([0, 180])

ax2.set_xlabel(r'$f$', fontsize=axis_label_fs)
ax2.set_ylabel(r'Power', fontsize=axis_label_fs)
ax2.set_xlim([freq_start, freq_end])

plt.show(fig)