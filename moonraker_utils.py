import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn


# ------------------------------------------------------------------------------------------
# Classes and constants
# ------------------------------------------------------------------------------------------

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


# A class to manage data for opening, updating, and closing a trade
class Trade:
    def __init__(self, symbol, open_price, open_time, open_tick, entry_tf, short, stop_loss):
        self.open_price = open_price
        self.open_time = open_time
        self.open_tick = open_tick
        self.entry_tf = entry_tf
        self.short = short
        self.stop_loss = stop_loss
        self.stopped_out = False
        self.closed = False
        self.close_price = None
        self.close_time = None
        self.close_tick = None
        self.color = None
        self.cur_capital = open_price
        self.pnl = 0.0
        self.pnl_percent = 0.0

    # Return the amount of capital at a given price point
    def get_capital_at_price(self, price):
        if self.short:
            return self.open_price + (self.open_price - price)
        else:
            return self.open_price - (self.open_price - price)

    # Update trade state
    def update(self, cur_price):
        self.cur_capital = self.get_capital_at_price(cur_price)
        self.pnl = self.cur_capital - self.open_price
        self.pnl_percent = self.pnl/self.open_price

    # Close the trade
    def close(self, close_price, close_time, close_tick, color, stopped_out):
        self.update(close_price)

        self.close_price = close_price
        self.close_time = close_time
        self.close_tick = close_tick
        self.closed = True
        self.color = color
        self.stopped_out = stopped_out


# ------------------------------------------------------------------------------------------
# Plotting helper functions
# ------------------------------------------------------------------------------------------

# Draw a labeled heatmap with the option to save to file
def draw_heatmap(row_labels, col_labels, data, title=None, filename=None, save=False, display = True, annot=None, xlabel=None, ylabel=None):
    # Create dataframe from row/column labels
    df = {}
    for i in range(0, len(col_labels)):
        df[col_labels[i]] = pd.Series(data[:, i], index=[str(x) for x in row_labels])
    df = pd.DataFrame(df)

    # Configure, draw, save heatmap
    sn.set(font_scale=0.7)
    palette = sn.diverging_palette(h_neg=10, h_pos=230, s=99, l=55, sep=3, as_cmap=True)
    ax = sn.heatmap(df, annot=annot, cmap=palette, center=0.00, cbar=False, fmt='')
    ax.figure.set_size_inches(6, 5.5)
    plt.title(title)
    plt.subplots_adjust(hspace=0)
    plt.subplots_adjust(left=0.1)
    plt.subplots_adjust(right=0.95)
    plt.subplots_adjust(bottom=0.03)
    plt.subplots_adjust(top=0.97)
    if save:
        plt.savefig(os.getcwd() + "/data/" + filename + ".png")
    if display:
        plt.show()
