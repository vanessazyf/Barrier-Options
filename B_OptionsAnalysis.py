import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import Analysis

def up_and_in_call(s, k, h, r, q, t, v):
    v_sqr_t = v * np.sqrt(t)
    lam = (r - q + v * v / 2) / (v * v)
    x1 = np.log(s / h) / v_sqr_t + lam * v_sqr_t
    y1 = np.log(h / s) / v_sqr_t + lam * v_sqr_t
    y = np.log(h * h / (s * k)) / v_sqr_t + lam * v_sqr_t

    term1 = s * norm.cdf(x1) * np.exp(-q * t)
    term2 = -k * np.exp(-r * t) * norm.cdf(x1 - v_sqr_t)
    term3 = -s * np.exp(-q * t) * (h / s) ** (2 * lam) * (norm.cdf(-y) - norm.cdf(-y1))
    term4 = k * np.exp(-r * t) * (h / s) ** (2 * lam - 2) * (norm.cdf(-y + v_sqr_t) - norm.cdf(-y1 + v_sqr_t))

    return term1 + term2 + term3 + term4

def up_and_out_call(s, k, h, r, q, t, v):
    f = s * np.exp((r - q) * t)
    return Analysis.black_option_price(f, k, t, v, 1) - up_and_in_call(s, k, h, r, q, t, v)

# this part of code won't get called by ipynb
if __name__ == "__main__":

    k = 100
    t = 1
    h = 130
    r = 0
    q = 0
    v = 0.2

    lower_bound = 10
    upper_bound = 130
    s = np.arange(lower_bound, upper_bound, 5)

    upAndInCall = []
    upAndOutCall = []
    european = []

    for i in range(len(s)):
        upAndInCall.append(up_and_in_call(s[i], k, h, r, q, t, v))
        upAndOutCall.append(up_and_out_call(s[i], k, h, r, q, t, v))
        f = s[i] * np.exp((r - q) * t)
        european.append(Analysis.black_option_price(f, k, t, v, 1))

    plt1, = plt.plot(s, upAndInCall, label="up and in call")
    plt2, = plt.plot(s, upAndOutCall, label="up and out call")
    plt3, = plt.plot(s, european, label="european call")

    plt.legend(handles=[plt1, plt2, plt3])
    plt.grid(True)
    plt.show()