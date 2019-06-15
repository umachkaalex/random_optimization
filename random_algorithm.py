import pandas as pd
import numpy as np
pd.options.mode.chained_assignment = None
pd.set_option('display.max_columns', 10)
np.set_printoptions(suppress=True)

# calculate financial metrics
def calc_fin_indicators(port_capital, start, end):
    # define empty dictionary as container for next calculations.
    fin_indicators = {}
    # calculcate the amount of years. it is needed to calculate CAGR
    pr_years = (end-start).days/365
    # calculcate portfolio return: portfolio value at the end of the period
    # devide on portfolio value at the start of the period.
    gain = port_capital[-1] / port_capital[0]
    # calculate CAGR
    CAGR = gain ** (1 / pr_years) - 1
    # calculate daily returns: portfolio value at the end of the current day
    # devide on portfolio value at the end of the previous day.
    daily_gain = np.diff(port_capital, axis=0) / port_capital[:-1]
    # calculate standard deviation as it is considered as portfolio risk.
    std = np.std(daily_gain, ddof=1)*np.sqrt(252)
    # calculate sharpe ration
    sharpe = CAGR / std
    # add financial parameters to dictionary
    fin_indicators['sharpe'] = sharpe
    fin_indicators['CAGR'] = CAGR
    fin_indicators['st_dev'] = std

    return fin_indicators

# To calculate financial metrics this function requires daily portfolio performance
# for the period defined by start and end dates. So let's write such function
def port_capital_flow(closes_data, st_cap, weights):
    # define the shape of array with closes
    m_shape = closes_data.shape
    # initialize the empty array to store the amount of shares
    num_shares_data = np.zeros(m_shape)
    # initialize the empty array to store the portfolio perfomance
    capital_data = np.zeros(m_shape)
    # start loop to calculate every day values of portfolio`s positions.
    for m in range(capital_data.shape[0]):
        if m == 0:
            # if this is the first day of period: the initial value of portfolio
            # equals the starting capital
            cur_cap = st_cap
            # distribute the starting capital between stocks using the list of
            # weights            
            capital_data[m, :] = weights*cur_cap            
            # calculate the number of shares (that we will hold during current 
            # period) based on distributed capital            
            num_shares_data[0, :] = capital_data[m, :]/closes_data[m, :]            
        else:
            # if this is not the first day of period: calculate portfolio
            # performance using the number of shares and daily stock closes 
            capital_data[m, :] = num_shares_data[0, :]*closes_data[m, :]
    
    # summarize performance of all opened positions (columns)     
    port_perform = np.sum(capital_data, axis=1)
    
    return port_perform

# Additionally we need one more function that will a little bit correct the optimized weights.
def correct_weights(opt_weights, corr_value=0.001):
  # set to zeros weights that less than certain value (very little weights - as 
  # in real stock market we would never buy 1 stock, if 1000 is normal size)
    idx_less = (opt_weights < corr_value)
    idx_over = (opt_weights >= corr_value)
    subs = np.sum(opt_weights[idx_less])    
    subs_s = subs/np.sum(idx_over*1)
  # distribute across non-zero stocks weights of those that set to zeros.
    opt_weights[idx_over] += subs_s    
    opt_weights[idx_less] = 0
    subs_s = 1-np.sum(opt_weights)
    opt_weights[idx_over] += subs_s/np.sum(idx_over*1)
    
    return opt_weights

def init_weights(data, random=True):
    # def shape of array of closes to create the same shape array of weights
    shape = data.shape[1]        
    if random:
        # create random weights
        weights = np.random.rand(shape).reshape(1, shape)
        weights = weights/np.sum(weights)
    else:
        # create equal weights
        weights = np.asarray([1/shape]*shape).reshape(1, shape)
        
    return weights

# Random optimization
def algo_random_ports(tickers, closes_data, num_of_ports, stocks_in_port,
                      start, end, print_info=True):
    # array to store values of label indicator calculated for each portfolio.
    # the values have to be added within the indexes of stocks (from "tickers" array)
    # that were included to portfolio.
    tickers_scores = np.zeros([1, len(tickers)])
    # array to store number of appearances of stocks in all created random portfolios.
    # +1 added within the indexes of stocks each time when certain stock is added to portfolio
    tickers_counts = np.ones([1, len(tickers)])

    # create loop for every stock
    for i in range(len(tickers)):
        # create list of indexes of all stock tickers
        cur_range = list(range(len(tickers)))
        # delete from range index of current stock (as it has to be added a priori)
        cur_range.remove(i)
        # start loop for creating p number of ports, where n number of random stocks plus current (i) stock
        for p in range(num_of_ports):
            # create n number of random indexes within range of number of stocks excluding
            # the index of "current" stock
            idx = np.random.choice(cur_range, stocks_in_port - 1, replace=False)
            # add index of "current" stock
            idx = np.append(idx, i)

            # slice data to get close prices of stocks from current portfolio
            data = closes_data[:, idx]
            # initialize equal weights
            weights = init_weights(data, random=False)
            # calculate current portfolio performance
            capital_flow = port_capital_flow(data, 10000, weights)
            # calculate desirable indicator
            fin_indicators = calc_fin_indicators(capital_flow, start, end)
            # update score and count arrays
            tickers_scores[0, idx] += fin_indicators['sharpe']
            tickers_counts[0, idx] += 1
    # create DataFrame to check the results
    rand_df = pd.DataFrame({'symbol': tickers,
                            'sharpe': tickers_scores.tolist()[0],
                            'number': tickers_counts.tolist()[0]})
    rand_df.to_csv('rand_df.csv', index=False)
    # create stocks rating
    tickers_scores = np.argsort(tickers_scores / tickers_counts)
    best_ind = tickers_scores[0, len(tickers) - stocks_in_port:]
    best_port = np.asarray(tickers)[best_ind]
    if print_info:
        print('Average appearance: ' + str(np.mean(tickers_counts)))
        print('StDev of appearance: ' + str(np.std(tickers_counts)))

    return best_port, best_ind

# Random optimization with risk constraint.
def algo_random_ports_risk(tickers, closes_data, num_of_ports, stocks_in_port, 
                           start, end, max_risk, print_info=True):
    
    tickers_scores = np.zeros([1, len(tickers)])     
    tickers_counts = np.ones([1, len(tickers)])
    # array to store values of risk calculated for each portfolio where current stock appeared.
    tickers_risk = np.zeros([1, len(tickers)])
    
    for i in range(len(tickers)):         
        cur_range = list(range(len(tickers)))        
        cur_range.remove(i)                
        for p in range(num_of_ports):            
            idx = np.random.choice(cur_range, stocks_in_port - 1, replace=False)            
            idx = np.append(idx, i)
                 
            data = closes_data[:, idx]           
            weights = init_weights(data, random=False)
            capital_flow = port_capital_flow(data, 10000, weights)            
            fin_indicators = calc_fin_indicators(capital_flow, start, end)
                        
            # update score, risk and count arrays
            tickers_scores[0, idx] += fin_indicators['sharpe']
            tickers_counts[0, idx] += 1            
            tickers_risk[0, idx] += fin_indicators['st_dev']
    
    rand_df_risk = pd.DataFrame({'symbol': tickers,
                                 'sharpe': tickers_scores.tolist()[0],
                                 'number': tickers_counts.tolist()[0],
                                 'st_dev': tickers_risk.tolist()[0]})    
    rand_df_risk.to_csv('rand_df_risk.csv', index=False)
    # count average portfolio risk for each stock appeared at that portfolios  
    tickers_aver_risk = tickers_risk / tickers_counts    
    # create mask to filter stocks that have average risk more than maximum acceptable    
    risk_idx = np.where(tickers_aver_risk < max_risk)
    # filter tickers with risk filter indexes    
    f_tickers = np.asarray(tickers).reshape(1, len(tickers))[:, risk_idx[1]][0]        
    # filter closes data with risk filter indexes
    f_closes_data = closes_data[:, risk_idx[1]]
    # check if number of tickers after filtering by risk is more then required 
    # number of stocks in portfolio
    if len(f_tickers)>stocks_in_port:
      # start random algorithm to define best portfolio allocation using limit_risk stocks
      best_port, best_ind = algo_random_ports(f_tickers, f_closes_data, num_of_ports, 
                                              stocks_in_port, pr_start_date,
                                              pr_end_date, print_info)
      # if the filter by risk return more stocks then required - return True
      return True, best_port, best_ind, f_tickers, f_closes_data
    else:
      # if the filter by risk return less stocks then required - return False
      best_port = ''
      best_ind = ''
      return False, best_port, best_ind, f_tickers, f_closes_data

# define elbow stop
def elbow_method(data, early_stop_k=1):
    # list to store 'elbow' values of each step
    elbow_signals = []

    # initialize early stop variable
    early_stop = 0

    # start calculation for each step starting from the second
    for i in range(1, len(data) - 1):
        # prevent division by zero
        if data[i - 1] != data[i]:
            # calculate current step elbow signal
            cur_signal = (data[i] - data[i + 1]) / (data[i - 1] - data[i])
            if i != 1:
                # check if current step signal lower then previous
                if cur_signal <= elbow_signals[-1]:
                    # if lower - add to list
                    elbow_signals.append(cur_signal)
                else:
                    # if not lower - check if early stop value doesn't equal base value
                    if early_stop != early_stop_k:
                        # if not - add 1 to early stop and store signal to list
                        early_stop += 1
                        elbow_signals.append(cur_signal)
                    else:
                        # if equal - stop calculation
                        break
            else:
                # if i equal 1 - simpy store signal because nothing to compare
                elbow_signals.append(cur_signal)
        else:
            # if data[i-1] == data[i] (denominator is zero)
            # check if early stop value doesn't equal base value
            if early_stop != early_stop_k:
                # if not add one to it
                early_stop += 1
            else:
                # if equal - stop calculation
                break
    # return the number of step at which we stop
    return i

# run loops with different number of stocks used in optimization
def random_optimization(tickers, data, random_ports, tickers_in_port,
                        pr_start_date, pr_end_date, risk_limit, print_info=False):


    print('\n    *****Random algorithm*****')
    all_tickers = []
    len_tickers = []
    elbow_signals = []
    for ports in random_ports:
        success, best_port_risk, best_ind_risk, \
        f_tickers, f_closes_data = algo_random_ports_risk(tickers, data,
                                                          ports, tickers_in_port,
                                                          pr_start_date, pr_end_date,
                                                          risk_limit, print_info)
        if success:
          all_tickers.extend(best_port_risk)
          all_tickers = list(set(all_tickers))
          len_tickers.append(len(all_tickers))
          if len(len_tickers) > 2:
            elbow_signals.append(elbow_method(len_tickers))
            if len(elbow_signals) > 1 and elbow_signals[-1] == elbow_signals[-2]:
              r_closes = f_closes_data[:, best_ind_risk]
              r_equal_weights = init_weights(r_closes, random=False)
              r_port_perform = port_capital_flow(r_closes, 100000, r_equal_weights)
              r_port_params = calc_fin_indicators(r_port_perform, pr_start_date, pr_end_date)
              return r_port_params, best_port_risk
        else:
          return 'There are less than required stocks after risk filtering', '_'

url = 'https://raw.githubusercontent.com/umachkaalex/stockmarket/master/pr_data_closes.csv'
# load previous month data
all_pr_data_closes = pd.read_csv(url)
# delete columns (stocks) with zero closes
all_pr_data_closes = all_pr_data_closes.replace(0, pd.np.nan).dropna(axis=1)
# create list of symbols 'Date' column
all_pr_tickers = all_pr_data_closes.columns.tolist()[:500]
# convert dataframes to numpy arrays without 'Date' column
all_pr_data_closes = all_pr_data_closes.values[:, :500]
# set start/end dates for previous and next periods
pr_start_date = pd.to_datetime('11/30/2017')
pr_end_date = pd.to_datetime('12/31/2017')
# the list of number of random ports to use in random algorithm
random_ports = [10, 20, 40, 70, 110, 160, 220, 290, 370, 460, 560, 670]

fin_results, portfolio = random_optimization(all_pr_tickers, all_pr_data_closes, random_ports,
                                             20, pr_start_date, pr_end_date, 0.15, print_info=False)
print(fin_results)
print(portfolio)
