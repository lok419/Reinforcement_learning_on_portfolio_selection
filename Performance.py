import numpy as np

def cummulative_return(day_return):
    return np.array(day_return).cumprod()

def sharpe_ratio(day_return):
    log_day_return = np.log(np.array(day_return))
    risk_free_rate = np.log(1.0148) / 60    # Three-month U.S. Treasury bill
    return np.sqrt(log_day_return.size) * (np.average(log_day_return) - risk_free_rate) / np.std(log_day_return)

def volatility(day_return):
    log_day_return = np.log(np.array(day_return))
    return np.std(log_day_return)

def max_drawdown(day_return):
    cumulative_return = cummulative_return(day_return)
    max_drawdown = 0
    peak = cumulative_return[0]
    for x in cumulative_return:
        if x > peak:
            peak = x
        daily_drawdown = (peak - x) / peak
        if daily_drawdown > max_drawdown:
            max_drawdown = daily_drawdown
    return max_drawdown