import os
import sys
from collections import defaultdict
import itertools as it
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import moonraker_utils


# ------------------------------------------------------------------------------------------
# Backtest config
# ------------------------------------------------------------------------------------------

tfs = ['120', '240', '480', '720', '1D', '3D']
tf_relabels = {'60':'1h', '120':'2h', '240':'4h', '480':'8h', '720':'12h', '1D':'1D', '3D':'3D', '1W':'1W'}

symbols = ['AAVEUSDT', 'ADAUSDT', 'AVAXUSDT', 'BALUSDT', 'BNBUSDT', 'DCRUSDT', 'DOGEUSDT',
           'DOTUSDT', 'EOSUSDT', 'ETHBTC', 'ETHUSDT', 'LINKUSDT', 'RUNEUSDT', 'RLCUSDT',
           'SOLUSDT', 'SUSHIUSDT', 'SXPUSDT', 'XMRUSDT', 'XRPUSDT', 'UNIUSDT', 'XHVUSDT',
           'YFIUSDT', 'BTCUSDT', 'LTCUSDT']

# Exit trades within this % of the meanline
trade_exit_threshold = 0.006

# Plotting options
#  trades: plots trades on price graph of the first symbol and timeframe
#  symbol stats: plots summary stats for each symbol as a table
#  pnl: scatter plots trade PNL against time in trade
#  cumulative: plots summary stats across all symbols as a table
plotting = 'stats'


# ------------------------------------------------------------------------------------------
# Load data
# ------------------------------------------------------------------------------------------

dfs = {}
for (symbol, tf) in it.product(symbols, tfs):
    dfs[(symbol, tf)] = pd.read_csv(os.getcwd() + "/data/moonraker/BINANCE_" + symbol + ", " + tf + ".csv")


# ------------------------------------------------------------------------------------------
# Collect backtest stats
# ------------------------------------------------------------------------------------------

# Run trading backtest for each symbol and timeframe
looking_for_exit = False
prev_close_tick = None
closed_trades = defaultdict(list)
for (symbol, tf) in it.product(symbols, tfs):
    df = dfs[(symbol, tf)]

    # Iterate through ticks
    cur_trade = None
    for tick in range(1, len(df['time'])):
        cur_time = df['time'][tick]
        cur_close_price = df['close'][tick]
        cur_high_price = df['high'][tick]
        cur_low_price = df['low'][tick]
        prev_close_price = df['close'][tick - 1]
        cur_meanline = df[meanline][tick]
        prev_meanline = df[meanline][tick - 1]
        cur_upper_band_lowest = df[upper_band_lowest][tick]
        cur_upper_band_top = df[upper_band_top][tick]
        cur_upper_band_bottom = df[upper_band_bottom][tick]
        cur_lower_band_highest = df[lower_band_highest][tick]
        cur_lower_band_top = df[lower_band_top][tick]
        cur_lower_band_bottom = df[lower_band_bottom][tick]
        cur_long_sl = df['Stops For Longs'][tick]
        cur_short_sl = df['Stops For Shorts'][tick]

        # Manage existing trade
        if cur_trade is not None:
            closest_price = None

            # Mean is between high and low -> exit exactly at mean
            if cur_low_price <= cur_meanline and cur_meanline <= cur_high_price:
                closest_price = cur_meanline

            # Current low price is above meanline
            # Happens when entry is on a wick with entry candle close above mean
            # Exits position at the close of the entry candle
            if cur_low_price > cur_meanline:
                closest_price = cur_low_price
                cur_trade.close(closest_price, cur_time, tick, 'orange', False)
                closed_trades[(symbol, tf)].append(cur_trade)
                cur_trade = None
                prev_close_tick = tick
                continue

            # Current high price is below the mean, try to exit there
            if cur_high_price < cur_meanline:
                closest_price = cur_high_price

            # If closest price is within the exit threshold, exit trade
            if np.abs(closest_price - cur_meanline)/cur_meanline <= trade_exit_threshold:
                cur_trade.close(closest_price, cur_time, tick, 'red', False)
                closed_trades[(symbol, tf)].append(cur_trade)
                cur_trade = None
                prev_close_tick = tick

            # If we haven't closed the trade on this candle, update the stop loss
            if cur_trade is not None:
                cur_trade.stop_loss = cur_long_sl

        # Open new trades
        if cur_trade is None and prev_close_tick != tick and cur_low_price <= cur_lower_band_bottom and cur_lower_band_bottom <= cur_high_price:
            cur_trade = moonraker_utils.Trade(symbol, cur_lower_band_bottom, cur_time, tick, tf, short=False, stop_loss=cur_long_sl)


# Plot PNL vs time-in-position
if plotting == 'pnl':
    tf_to_int = {tf:tfs.index(tf) for tf in tfs}
    tf_to_color = {'120':'navy', '240':'darkviolet', '480':'slateblue', '720':'lightseagreen', '1D':'teal', '3D':'cadetblue'}
    for tf in tfs:
        PNLs = []
        durations = []
        colors = []
        for i in range(len(symbols)):
            ind = (symbols[i], tf)

            PNLs += [trade.pnl_percent for trade in closed_trades[ind]]
            durations += [(trade.close_time - trade.open_time)/(60*60*24) for trade in closed_trades[ind]]
            colors += [tf_to_color[trade.entry_tf] for trade in closed_trades[ind]]

        plt.scatter(durations, PNLs, s=10, c=colors, alpha=0.6, cmap='brg', label=tf_relabels[tf])

    plt.grid()
    plt.legend()
    plt.title("PNL vs time in trade")
    plt.subplots_adjust(hspace=0)
    plt.subplots_adjust(left=0.1)
    plt.subplots_adjust(right=0.95)
    plt.subplots_adjust(bottom=0.1)
    plt.subplots_adjust(top=0.9)
    plt.show()


# Plot trades on price graph for the first symbol and timeframe
elif plotting == 'trades':
    tick_range = [0, 2000]
    plot_tf = tfs[0]
    plot_symbol = symbols[0]

    fig, ax = plt.subplots()
    fig.set_size_inches(12, 6)

    incr = (tick_range[1] - tick_range[0])//10

    # Allow left-right scroll through price graph
    def press(event):
        sys.stdout.flush()
        ax.cla()

        # Move range according to keypress
        if event != '~' and event.key == 'left':
            if tick_range[0] != 0:
                tick_range[0] = tick_range[0] - incr
                tick_range[1] = tick_range[1] - incr
        elif event != '~' and event.key == 'right':
            if tick_range[1] <= len(dfs[(plot_symbol, plot_tf)]['close']):
                tick_range[0] = tick_range[0] + incr
                tick_range[1] = tick_range[1] + incr

        # Plot trades
        trades_to_plot_open = []
        trades_to_plot_close = []
        colors = []
        for x in closed_trades[(plot_symbol, plot_tf)]:
            if x.open_tick >= tick_range[0] and x.open_tick <= tick_range[1]:
                trades_to_plot_open.append([x.open_tick, x.open_price])
            if x.close_tick >= tick_range[0] and x.close_tick <= tick_range[1]:
                trades_to_plot_close.append([x.close_tick, x.close_price])
                colors.append(x.color)

        trades_to_plot_open = np.array(trades_to_plot_open)
        trades_to_plot_close = np.array(trades_to_plot_close)
        if len(trades_to_plot_open) > 0:
            ax.scatter(trades_to_plot_open[:, 0], trades_to_plot_open[:, 1], c='green', marker='o')
        if len(trades_to_plot_close) > 0:
            ax.scatter(trades_to_plot_close[:, 0], trades_to_plot_close[:, 1], c=colors, marker='x')

        # Plot price and trade signal scaled to price
        ax.plot(dfs[(plot_symbol, plot_tf)]['close'].iloc[tick_range[0]:tick_range[1]], c='black', label='price')
        ax.plot(dfs[(plot_symbol, plot_tf)][meanline].iloc[tick_range[0]:tick_range[1]], c='red', label='price', linewidth=0.5)
        ax.plot(dfs[(plot_symbol, plot_tf)][lower_band_highest].iloc[tick_range[0]:tick_range[1]], c='black', label='price', linewidth=0.5)
        plt.draw()

    fig.canvas.mpl_connect('key_press_event', press)
    press('~')
    plt.show()


# Plot summary stats for each symbol as a table
elif plotting == 'symbol stats':
    # Compute states from closed trade list
    expected_returns = np.zeros((len(symbols), len(tfs)))
    num_trades = np.zeros((len(symbols), len(tfs)))
    num_trades_per_week = np.zeros((len(symbols), len(tfs)))
    num_winners = np.zeros((len(symbols), len(tfs)))
    proportion_winners = np.zeros((len(symbols), len(tfs)))
    avg_trade_len = np.zeros((len(symbols), len(tfs)))
    for i in range(len(symbols)):
        for j in range(len(tfs)):
            ind = (symbols[i], tfs[j])

            expected_returns[i,j] = np.mean([trade.pnl_percent for trade in closed_trades[ind]])
            num_trades[i,j] = len(closed_trades[ind])
            num_trades_per_week[i,j] = len(closed_trades[ind])/((dfs[ind]['time'].iloc[-1] - dfs[ind]['time'].iloc[0])/(7*24*60*60))
            num_winners[i,j] = np.sum([trade.pnl_percent > 0 for trade in closed_trades[ind]])
            proportion_winners[i,j] = np.sum([trade.pnl_percent > 0 for trade in closed_trades[ind]]) / len(closed_trades[ind])
            avg_trade_len[i,j] = np.mean([(trade.close_time - trade.open_time)/(60*60*24) for trade in closed_trades[ind]])

    # Plot stats as a heatmap
    row_labels = [x for x in symbols]
    col_labels = [tf_relabels[tf] for tf in tfs]
    data = proportion_winners - 0.55

    annotations = [["%.1f%%, %d/%d ~ %.1f%%, %.1f, %.1fd" % (expected_returns[k,j]*100, num_winners[k,j], num_trades[k,j], proportion_winners[k,j]*100, num_trades_per_week[k,j], avg_trade_len[k,j]) for j in range(len(tfs))] for k in range(len(symbols))]
    annotations = np.asarray(strings).reshape(len(symbols), len(tfs))

    title = "Long at lowest line of bottom band, exit at meanline"
    ylabel = "Symbol"
    xlabel = "TF"
    moonraker_utils.draw_heatmap(row_labels, col_labels, data, title=title, filename=None, save=False, display=True, annot=annotations, xlabel=xlabel, ylabel=ylabel)


# Plot cumulative stats across all symbols
elif plotting == 'cumulative':
    # Compute stats from closed trade list
    expected_returns = np.zeros((1, len(tfs)))
    num_trades = np.zeros((1, len(tfs)))
    num_trades_per_week = np.zeros((1, len(tfs)))
    num_winners = np.zeros((1, len(tfs)))
    proportion_winners = np.zeros((1, len(tfs)))
    avg_trade_len = np.zeros((1, len(tfs)))
    for j in range(len(tfs)):
        expected_returns[0,j] = np.mean([trade.pnl_percent for s in symbols for trade in closed_trades[(s, tfs[j])]])
        num_trades[0,j] = np.sum([len(closed_trades[(s, tfs[j])]) for s in symbols])
        num_trades_per_week[0,j] = np.sum([len(closed_trades[(s, tfs[j])])/((dfs[(s, tfs[j])]['time'].iloc[-1] - dfs[(s, tfs[j])]['time'].iloc[0])/(7*24*60*60)) for s in symbols])
        num_winners[0,j] = np.sum([trade.pnl_percent > 0 for s in symbols for trade in closed_trades[(s, tfs[j])]])
        proportion_winners[0,j] = np.sum([trade.pnl_percent > 0  for s in symbols for trade in closed_trades[(s, tfs[j])]]) / np.sum([len(closed_trades[(s, tfs[j])]) for s in symbols])
        avg_trade_len[0,j] = np.mean([(trade.close_time - trade.open_time)/(60*60*24) for s in symbols for trade in closed_trades[(s, tfs[j])]])

    # Plot stats as a heatmap
    row_labels = ['']
    col_labels = [tf_relabels[tf] for tf in tfs]
    data = proportion_winners - 0.55

    annotations = [["%.1f%%, %d/%d ~ %.1f%%, %.1f, %.1fd" % (expected_returns[0,j]*100, num_winners[0,j], num_trades[0,j], proportion_winners[0,j]*100, num_trades_per_week[0,j], avg_trade_len[0,j]) for j in range(len(tfs))]]
    annotations = np.asarray(strings).reshape(1, len(tfs))

    title = "Long at lowest line of bottom band, exit at meanline, all coins"
    ylabel = "Symbol"
    xlabel = "TF"
    moonraker_utils.draw_heatmap(row_labels, col_labels, data, title=title, filename=None, save=False, display=True, annot=annotations, xlabel=xlabel, ylabel=ylabel)


