import os
from collections import defaultdict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup


# Meaningful labels for the moonraker lines
meanline = 'Plot'
meanline_plus = 'Plot.1'
meanline_minus = 'Plot.2'
upper_band_lowest = 'Plot.5'
upper_band_top = 'Plot.10'
upper_band_bottom = 'Plot.9'
lower_band_highest = 'Plot.7'
lower_band_top = 'Plot.11'
lower_band_bottom = 'Plot.12'


# Load tradingview ticker data for binance symbols
tf = '60'
symbols = ['AAVEUSDT', 'ADAUSDT', 'AVAXUSDT', 'BALUSDT', 'BNBUSDT', 'DCRUSDT', 'DOGEUSDT',
           'DOTUSDT', 'EOSUSDT', 'ETHBTC', 'ETHUSDT', 'LINKUSDT', 'RUNEUSDT', 'RLCUSDT',
           'SOLUSDT', 'SUSHIUSDT', 'SXPUSDT', 'XMRUSDT', 'XRPUSDT', 'UNIUSDT', 'XHVUSDT',
           'YFIUSDT', 'BTCUSDT', 'LTCUSDT']
dfs = {s:pd.read_csv(os.getcwd() + "/data/moonraker/BINANCE_" + s + ", " + tf + ".csv") for s in symbols}

# Time intervals to collect stats over
stats_time_intervals = [1,2,4,8,12,24,2*24,3*24]

# As a fraction of the distance between the meanline and upper band bottom
bin_size = 0.025

# Max distance from meanline to collect stats, in units of (upper band bottom - meanline)
distr_max = 5.0


# Iterate through symbols, binning price and recording stats
stats_bins = np.arange(-distr_max, distr_max + bin_size, bin_size) + bin_size/2
meanline_bin = [x < 0 for x in stats_bins].index(False) - 1
bins_per_percent = 1/(10*bin_size)
prob_mean_reversion = np.zeros((len(stats_bins), len(stats_time_intervals)))
for symbol in symbols:
    df = dfs[symbol]

    # Bin price
    price_in_bins = np.zeros((df.shape[0], len(stats_bins)))
    num_binned = 0
    for tick in range(df.shape[0]):
        # Skip ticks where the indicator has yet to render
        if np.isnan(df[meanline][tick]):
            continue

        close = df['close'][tick]
        mean_price = df[meanline][tick]
        upper_band_bottom_dist = df[upper_band_lowest][tick] - mean_price
        cur_stats_bins = stats_bins*upper_band_bottom_dist  + mean_price

        bin_found = False
        for i in range(1, len(stats_bins)):
            if cur_stats_bins[i-1] <= close and close < cur_stats_bins[i]:
                price_in_bins[tick, i-1] = 1
                bin_found = True
                break
            elif cur_stats_bins[len(stats_bins)-1] <= close and close < cur_stats_bins[len(stats_bins)-1] + bin_size:
                price_in_bins[tick, i] = 1
                bin_found = True

        # Price is outside the bin range, shouldn't happen often
        if not bin_found:
            print('price out of bin range')
            if close >= cur_stats_bins[-1]:
                price_in_bins[tick, -1] = 1
            else:
                price_in_bins[tick, 0] = 1
        else:
            num_binned += 1

    # Collect mean reversion stats for each interval
    prob_further_from_mean = np.zeros((len(stats_bins), len(stats_time_intervals)))
    prob_closer_to_mean = np.zeros((len(stats_bins), len(stats_time_intervals)))
    for tick in range(df.shape[0] - max(stats_time_intervals)):
        # Skip ticks where the indicator has yet to render
        if np.isnan(df[meanline][tick]):
            continue

        close = df['close'][tick]
        mean_price = df[meanline][tick]
        cur_bin = list(price_in_bins[tick,:]).index(1)

        for i in range(len(stats_time_intervals)):
            interval = stats_time_intervals[i]
            lookforward_bin = list(price_in_bins[tick + interval,:]).index(1)

            # Record whether price moved towards or away from the meanline
            if cur_bin == meanline_bin:
                prob_further_from_mean[cur_bin, i] += 1
                prob_closer_to_mean[cur_bin, i] += 1
            else:
                if close >= mean_price:
                    if lookforward_bin == cur_bin:
                        if df['close'][tick + interval] >= df['close'][tick]:
                            prob_further_from_mean[cur_bin, i] += 1
                        else:
                            prob_closer_to_mean[cur_bin, i] += 1
                    elif lookforward_bin > cur_bin:
                        prob_further_from_mean[cur_bin, i] += 1
                    else:
                        prob_closer_to_mean[cur_bin, i] += 1
                else:
                    if lookforward_bin == cur_bin:
                        if df['close'][tick + interval] <= df['close'][tick]:
                            prob_further_from_mean[cur_bin, i] += 1
                        else:
                            prob_closer_to_mean[cur_bin, i] += 1
                    elif lookforward_bin < cur_bin:
                        prob_further_from_mean[cur_bin, i] += 1
                    else:
                        prob_closer_to_mean[cur_bin, i] += 1

    # Normalize mean reversion probabilities
    for i in range(len(stats_time_intervals)):
        for bin in range(len(stats_bins)):
            if prob_closer_to_mean[bin, i] == 0 and prob_further_from_mean[bin, i] == 0:
                continue

            prob_closer_to_mean[bin, i] /= prob_closer_to_mean[bin, i] + prob_further_from_mean[bin, i]
            prob_further_from_mean[bin, i] /= prob_closer_to_mean[bin, i] + prob_further_from_mean[bin, i]

    prob_mean_reversion += prob_closer_to_mean

prob_mean_reversion /= len(symbols)
prob_mean_reversion = np.flip(prob_mean_reversion, axis=0)


# Plot probability of mean reversion as time elapses starting from a given bin
plotted_lookforwards = stats_time_intervals
interval_to_label = {1:"1h", 2:'2h', 4:'4h', 8:'8h', 12:'12h', 24:'1D', 48:'2D', 72:'3D'}
for i in range(len(stats_time_intervals)):
    linewidth = ((i+1)/(len(stats_time_intervals)-1))/2 + 0.25
    plt.plot(prob_mean_reversion[:,i], label=interval_to_label[stats_time_intervals[i]], color='black', linewidth=linewidth)

# Configure x axis labels
tick_percents = [-25, -12.5, -10, 0, 10, 12.5, 25]
xticks = [bins_per_percent*x + meanline_bin for x in tick_percents]
xticklabels = ['+25%', '+12.5%', '+10%', 'meanline', '-10%', '-12.5%', '-25%']
plt.gca().set_xticks(xticks)
plt.gca().set_xticklabels(xticklabels)
for label in plt.gca().get_xticklabels():
    label.set_rotation(90)

# Configure y axis labels
yticks = [0.1*x for x in range(11)]
yticklabels = [" %d%%"%(int(x*100)) for x in yticks]
plt.gca().set_yticks(yticks)
plt.gca().set_yticklabels(yticklabels)
for label in plt.gca().get_yticklabels():
    label.set_rotation(90)
    label.set_verticalalignment('center')
plt.legend()
plt.show()

