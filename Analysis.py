import numpy as np
from scipy.stats import norm
from scipy import optimize
import matplotlib.pyplot as plt


# undiscounted black option price
def black_option_price(f,  # forward, double
                       k,  # strike, double
                       t,  # time to maturity, double
                       v,  # implied volatility, double
                       c_or_p  # call (1) or put (-1), integer
                       ):

    d_1 = (np.log(f/k)+0.5*v*v*t)/(v*np.sqrt(t))
    d_2 = d_1 - v*np.sqrt(t)
    if c_or_p == 1:
        return f * norm.cdf(d_1) - k * norm.cdf(d_2)
    elif c_or_p == -1:
        return k * norm.cdf(-d_2) - f * norm.cdf(-d_1)
    else:
        raise ValueError('c_or_p is expected to be 1 for call or -1 for put.')


# undiscounted black option vega
def black_option_vega(f,  # forward, double
                      k,  # strike, double
                      t,  # time to maturity, double
                      v   # implied volatility, double
                      ):

    d_1 = (np.log(f / k) + 0.5 * v * v * t) / (v * np.sqrt(t))
    return f * norm.pdf(d_1) * np.sqrt(t)


# compute black implied volatility
def black_implied_vol(p,  # option price, double
                      f,  # forward, double
                      k,  # strike, double
                      t,  # time to maturity, double
                      c_or_p,  # call (1) or put (-1), integer
                      init_guess = 0.2 # initial guess
                      ):

    f_ivol = lambda x: black_option_price(f, k, t, x, c_or_p) - p
    f_vega = lambda x: black_option_vega(f, k, t, x)
    return optimize.newton(f_ivol, init_guess, f_vega)


# undiscounted Bachelier option
def bachelier_option_price(f,  # forward, double
                           k,  # strike, double
                           t,  # time to maturity, double
                           v,  # bachelier implied volatility, double
                           c_or_p  # call (1) or put (-1), integer
                           ):
    v_sqrt_t = v * np.sqrt(t)
    d = (f - k) / v_sqrt_t

    if c_or_p == 1 :
        return (f - k) * norm.cdf(d) + v_sqrt_t * norm.pdf(d)
    elif c_or_p == -1 :
        return (k - f) * norm.cdf(-d) + v_sqrt_t * norm.pdf(-d)
    else:
        raise ValueError('c_or_p is expected to be 1 for call or -1 for put.')


# undiscounted shifted lognormal option price
# dF = (F+shift) v_sln dW
def sln_option_price(f,  # forward, double
                     k,  # strike, double
                     t,  # time to maturity, double
                     v_sln,  # sln implied volatility, double
                     shift,  # the shift
                     c_or_p  # call (1) or put (-1), integer
                     ):
    return black_option_price(f + shift, k + shift, t, v_sln, c_or_p)


# undiscounted black option vega
def sln_option_vega(f,  # forward, double
                    k,  # strike, double
                    t,  # time to maturity, double
                    v_sln,  # sln implied volatility, double
                    shift  # the shift
                    ):
    return black_option_vega(f+shift, k+shift, t, v_sln)


# compute sln implied volatility
def sln_implied_vol(p,  # option price, double
                    f,  # forward, double
                    k,  # strike, double
                    t,  # time to maturity, double
                    c_or_p,  # call (1) or put (-1), integer
                    shift,  # the shift
                    init_guess=0.2  # initial guess
                    ):

    f_ivol = lambda x: sln_option_price(f, k, t, x, shift, c_or_p) - p
    f_vega = lambda x: sln_option_vega(f, k, t, x, shift)
    return optimize.newton(f_ivol, init_guess, f_vega)


# this part of code won't get called by ipynb
if __name__ == "__main__":

    f = 100
    t = 1
    shift = 30

    lower_bound = 50
    upper_bound = 150
    ks = np.arange(lower_bound, upper_bound, 5)

    black_atm_ivol = 0.2
    bach_atm_ivol = black_atm_ivol * f

    atm_call_price = black_option_price(f, f, t, black_atm_ivol, 1)

    # find the shifted lognormal vol for the given shift to calbirate to the ATM call price
    sln_atm_ivol = sln_implied_vol(atm_call_price, f, f, t, 1, shift)

    # compute the call option prices
    black_call = [black_option_price(f, ks[i], t, black_atm_ivol, 1) for i in range(len(ks))]
    bach_call = [bachelier_option_price(f, ks[i], t, bach_atm_ivol, 1) for i in range(len(ks))]
    sln_call = [sln_option_price(f, ks[i], t, sln_atm_ivol, shift, 1) for i in range(len(ks))]

    # compute the implied vol from the 3 models
    black_ivol = [black_implied_vol(black_call[i], f, ks[i], t, 1, black_atm_ivol) for i in range(len(ks))]
    bach_ivol = [black_implied_vol(bach_call[i], f, ks[i], t, 1, black_atm_ivol) for i in range(len(ks))]
    sln_ivol = [black_implied_vol(sln_call[i], f, ks[i], t, 1, black_atm_ivol) for i in range(len(ks))]

    plt1, = plt.plot(ks, black_ivol, label="black ")
    plt2, = plt.plot(ks, bach_ivol, label="bachelier ")
    plt3, = plt.plot(ks, sln_ivol, label="sln shift")

    plt.legend(handles=[plt1, plt2, plt3])
    plt.grid(True)
    plt.show()
