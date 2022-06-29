
import threading
import time
from datetime import datetime
import pandas as pd
import numpy as np
from binance.client import Client

import ccxt
import talib

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from Root import Keys

import numba
from numba import jit, njit

CANDLE_DURATION_IN_MIN = 5

RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30

CCXT_TICKER_NAME = 'ETH/USDT'
TRADING_TICKER_NAME = 'ethusdt'

INVESTMENT_AMOUNT_DOLLARS = 10
HOLDING_QUANTITY = 0

exchange = ccxt.binance()
bc_client = Client(api_key=Keys.API_key, api_secret=Keys.API_secret)



def fetch_data(ticker):
    start = time.time()
    global exchange
    bars, ticker_df = None, None

    try:
        bars = exchange.fetch_ohlcv(ticker, timeframe=f'{CANDLE_DURATION_IN_MIN}m', limit=100)

    except:
        print(f"Error in fetching data from the exchange:{ticker}")

    #print(ticker)

    if bars is not None:
        ticker_df = pd.DataFrame(bars[:-1], columns=['at', 'open', 'high', 'low', 'close', 'vol'])
        ticker_df['Date'] = pd.to_datetime(ticker_df['at'], unit='ms')
        ticker_df['symbol'] = ticker

    end = time.time()

    print("1st", end - start)
    return ticker_df




@jit(parallel=True)
def get_trade_recommendation(ticker_df):
    start = time.time()
    macd_result = 'WAIT'
    final_result = 'WAIT'

    macd, signal, hist = talib.MACD(ticker_df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
    last_hist = hist.iloc[-1]
    prev_hist = hist.iloc[-2]
    #print(hist)
    #print(last_hist)
    #print(prev_hist)

    if not np.isnan(prev_hist) and not np.isnan(last_hist):
        # If hist value has changed from negative to positive or vice versa, it indicates a crossover
        macd_crossover = (abs(last_hist + prev_hist)) != (abs(last_hist) + abs(prev_hist))

        if macd_crossover:
            macd_result = 'BUY' if last_hist > 0 else 'SELL'

        #print(macd_crossover)
        #print(macd_result)
    rsi = talib.RSI(ticker_df['close'], RSI_PERIOD)

    if macd_result != 'WAIT':
        rsi = talib.RSI(ticker_df['close'], RSI_PERIOD)

        # Consider last 3 RSI values
        last_rsi_values = rsi.iloc[-3:]
        #print(rsi)

        if last_rsi_values.min() <= RSI_OVERSOLD:
            final_result = 'BUY'
        elif last_rsi_values.max() >= RSI_OVERBOUGHT:
            final_result = 'SELL'
    plot_macd(ticker_df, macd, signal, hist, rsi)
   #plot_rsi(ticker_df, rsi)

    end = time.time()

    print("2nd", end - start)

    return final_result




def execute_trade(trade_rec_type, trading_ticker):
    start = time.time()
    global bc_client, HOLDING_QUANTITY
    order_placed = False
    side_value = 'buy' if (trade_rec_type == "BUY") else 'sell'
    try:
        ticker_price_response = bc_client.create_order("ticker", {"symbol": trading_ticker})
        #print(ticker_price_response)
        if (ticker_price_response[0] in [200, 201]):
            current_price = float(ticker_price_response[1]['lastPrice'])

            scrip_quantity = round(INVESTMENT_AMOUNT_DOLLARS/current_price,5) if trade_rec_type == "BUY" else HOLDING_QUANTITY
            print(f"PLACING ORDER {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}: "
                  f"{trading_ticker}, {side_value}, {current_price}, {scrip_quantity}, {int(time.time() * 1000)} ")

            order_response = bc_client.create_order("create_order",
                                        {"symbol": trading_ticker, "side": side_value, "type": "limit",
                                         "price": current_price, "quantity": scrip_quantity,
                                         "recvWindow": 10000, "timestamp": int(time.time() * 1000)})

            print(f"ORDER PLACED")
            HOLDING_QUANTITY = scrip_quantity if trade_rec_type == "BUY" else HOLDING_QUANTITY
            order_placed = True
    except:
        print(f"\nALERT!!! UNABLE TO COMPLETE ORDER")

    end = time.time()

    print("3rd", end - start)

    return order_placed


@jit(parallel=True)
def run_bot_for_ticker(ccxt_ticker, trading_ticker):
    start = time.time()

    currently_holding = False
    while 1:
        # STEP 1: FETCH THE DATA
        ticker_data = fetch_data(ccxt_ticker)
        #print(ticker_data)
        if ticker_data is not None:
            # STEP 2: COMPUTE THE TECHNICAL INDICATORS & APPLY THE TRADING STRATEGY
            trade_rec_type = get_trade_recommendation(ticker_data)
            print(f'{datetime.now().strftime("%d/%m/%Y %H:%M:%S")}  TRADING RECOMMENDATION: {trade_rec_type}')

            # STEP 3: EXECUTE THE TRADE
            if (trade_rec_type == 'BUY' and not currently_holding) or \
                (trade_rec_type == 'SELL' and currently_holding):
                print(f'Placing {trade_rec_type} order')
                trade_successful = execute_trade(trade_rec_type, trading_ticker)
                currently_holding = not currently_holding if trade_successful else currently_holding

            time.sleep(CANDLE_DURATION_IN_MIN*60)
        else:
            print(f'Unable to fetch ticker data - {ccxt_ticker}. Retrying!!')
            time.sleep(10)

    end = time.time()

    print("4th", end - start)

def plot_macd(ticker_df, macd, signal, hist, rsi):
    fig = make_subplots(rows=3, cols=1)

    candlestick = go.Candlestick(x=ticker_df.index,
                                 open=ticker_df['open'],
                                 high=ticker_df['high'],
                                 low=ticker_df['low'],
                                 close=ticker_df['close'], name='daily candle')

    positive_hist = hist[hist > 0]
    negative_hist = hist[hist < 0]

    macd_line = go.Scatter(x=macd.index, y=macd, name='macd', line_color='blue')
    signal_line = go.Scatter(x=signal.index, y=signal, name='signal', line_color='orange')
    RSI_line = go.Scatter(x=rsi.index, y=rsi, name='rsi', line_color='blue')
    pos_hist_bar = go.Bar(x=positive_hist.index, y=positive_hist, name='hist', marker={'color': 'green'})
    neg_hist_bar = go.Bar(x=negative_hist.index, y=negative_hist, name='hist', marker={'color': 'red'})

    fig.add_trace(candlestick, row=1, col=1)
    fig.add_trace(macd_line, row=2, col=1)
    fig.add_trace(signal_line, row=2, col=1)
    fig.add_trace(pos_hist_bar, row=2, col=1)
    fig.add_trace(neg_hist_bar, row=2, col=1)
    fig.add_trace(RSI_line, row=3, col=1)
    fig.add_hline(y=70, line_color="green", line_dash="dash", row=3, col=1)
    fig.add_hline(y=30, line_color="green", line_dash="dash", row=3, col=1)


    fig.update_layout(title_text=f'MACD-RSI', title_x=0.5,
                      xaxis_type='category', xaxis_rangeslider_visible=False,
                      xaxis_showticklabels=False,
                      xaxis2_type='category', paper_bgcolor="grey")

    fig.show()

#ef plot_rsi(ticker_df, rsi):
   # fig = px.line(ticker_df, x=rsi.index, y=rsi, title='RSI')
   # fig.add_hline(y=70, fillcolor="red")
   # fig.add_hline(y=30, fillcolor="red")
   # fig.show()


'''t = threading.Thread(target=fetch_data)
t1 = threading.Thread(target=get_trade_recommendation)
t2 = threading.Thread(target=execute_trade)
t3 = threading.Thread(target=run_bot_for_ticker)

t.start()
t1.start()
t2.start()
t3.start()

t.join()
t1.join()
t2.join()
t3.join()
'''


start = time.time()
run_bot_for_ticker(CCXT_TICKER_NAME, TRADING_TICKER_NAME)
end = time.time()

print(end-start)

