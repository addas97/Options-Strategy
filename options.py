import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.stats import norm
import matplotlib.pyplot as plt
import time

# THURS + FRI ON EXPIRATION WEEKS (OPEX, VIX, Monthly) - AROUND EOD - MARKET RALLIES OR SHORTS bc of gamma flip - 
# If (-) GEX to (+) GEX - buy underlying toward (~1pm) EOD if shorted in AM
# If (+) GEX to (-) GEX - sell underlying toward EOD if long in AM

# === CONSTANTS ===
RISK_FREE_RATE = 0.0426
MAX_OPTION_EXPIRATIONS = 4

def get_stock_data(ticker):

    stock = yf.Ticker(ticker)
    current_price = stock.info['currentPrice']
    options_data = []
    options_expirations = stock.options
    
    today = datetime.today().weekday()

    if today <= 2 or today >= 5:
        for _ in options_expirations[0:4]:
            options = stock.option_chain(_)
            calls = options.calls
            puts = options.puts
            options_data.append({'Exp': _, 'Calls': calls, 'Puts': puts})

    else:
        for _ in options_expirations[1:5]:
            options = stock.option_chain(_)
            calls = options.calls
            puts = options.puts
            options_data.append({'Exp': _, 'Calls': calls, 'Puts': puts})

    df = pd.DataFrame(options_data)

    return current_price, df

# === CALL RESISTANCE LEVEL ===
def call_option_data(df, current_price):
    
    call_df = df[["Exp", "Calls"]]
    final_call_df = pd.DataFrame()
    price_13_pct_itm = current_price * .87
    price_13_pct_otm = current_price * 1.13
    
    for call_option_per_exp in call_df["Calls"]:
        itm_calls = call_option_per_exp.loc[(call_option_per_exp['strike'] >= price_13_pct_itm) & (call_option_per_exp['inTheMoney'] == True)]
        otm_calls = call_option_per_exp.loc[(price_13_pct_otm >= call_option_per_exp['strike']) & (call_option_per_exp['inTheMoney'] == False)]
        combined_df = pd.concat([itm_calls, otm_calls], ignore_index = True)
        final_call_df = pd.concat([final_call_df, combined_df], ignore_index = True)

    final_call_df["Expiration"] = final_call_df['contractSymbol'].apply(lambda symbol: ''.join(char for char in symbol if char in '0123456789')[:6])
    final_call_df["Expiration"] = final_call_df['Expiration'].apply(lambda exp: '20' + '-'.join(exp[i:i + 2] for i in range(0, len(exp), 2)))

    return final_call_df

def max_OI(final_call_df):
    
    max_OI = final_call_df.groupby("Expiration", as_index = False)['openInterest'].max()
    oi_lst = []

    for i in max_OI['openInterest']:
        oi_lst.append(i)
    
    highest_OI = final_call_df[final_call_df['openInterest'].isin(oi_lst)]
    strikes = [strike for strike in highest_OI['strike']]
    
    return max(list(set(strikes)))

# === PUT SUPPORT LEVEL ===
def put_option_data(df, current_price):
    
    put_data = df[["Exp", "Puts"]]
    final_put_df = pd.DataFrame()
    price_13_pct_itm = current_price * 1.13
    price_13_pct_otm = current_price * .87

    for row in put_data['Puts']:
        itm_puts = row.loc[(row['strike'] <= price_13_pct_itm) & (row['inTheMoney'] == True)]
        otm_puts = row.loc[(price_13_pct_otm <= row['strike']) & (row['inTheMoney'] == False)]
        combined_df = pd.concat([itm_puts, otm_puts], ignore_index=True)
        final_put_df = pd.concat([final_put_df, combined_df], ignore_index=True)
    
    final_put_df["Expiration"] = final_put_df['contractSymbol'].apply(lambda symbol: ''.join(char for char in symbol if char in '0123456789')[:6])    
    final_put_df["Expiration"] = final_put_df['Expiration'].apply(lambda exp: '20' + '-'.join(exp[i:i + 2] for i in range(0, len(exp), 2)))

    return final_put_df

def max_OI_put(final_put_df):
    
    max_OI = final_put_df.groupby('Expiration', as_index = False)['openInterest'].max()
    oi_lst = []

    for oi in max_OI['openInterest']:
        oi_lst.append(oi)

    highest_put_oi = final_put_df[final_put_df['openInterest'].isin(oi_lst)]
    strikes = [strike for strike in highest_put_oi['strike']]

    return max(list(set(strikes)))

# === GAMMA LEVELS ===
def days_exp(final_call_df, final_put_df):
    
    final_call_df["TTM"] = (pd.to_datetime(final_call_df["Expiration"]) - datetime.now()).dt.days / 365.0
    final_call_df["TTM"] = final_call_df["TTM"].astype(float)
    final_call_df["TTM"] = final_call_df["TTM"].apply(lambda x: max(x, 0.00000001))
    final_put_df["TTM"] = (pd.to_datetime(final_put_df["Expiration"]) - datetime.now()).dt.days / 365.0
    final_put_df["TTM"] = final_put_df["TTM"].astype(float)
    final_put_df["TTM"] = final_put_df["TTM"].apply(lambda x: max(x, 0.00000001))

    return final_call_df, final_put_df

def calc_gamma(final_call_df, final_put_df, current_price, rfr):
    
    final_call_df["lastPrice"] = final_call_df["lastPrice"].astype(float)
    final_call_df["strike"] = final_call_df["strike"].astype(float)
    final_put_df["lastPrice"] = final_put_df["lastPrice"].astype(float)
    final_put_df["strike"] = final_put_df["strike"].astype(float)

    # Gamma Calculation
    final_call_df["d1"] = ((np.log(current_price / final_call_df["strike"]) + (rfr + 0.5 * final_call_df["impliedVolatility"] ** 2) * final_call_df['TTM']) / (final_call_df["impliedVolatility"] * np.sqrt(final_call_df['TTM'])))
    final_call_df["Call Gamma"] = norm.pdf(final_call_df['d1']) / (current_price * final_call_df['impliedVolatility'] * np.sqrt(final_call_df['TTM']))

    final_put_df["d1"] = ((np.log(current_price / final_put_df["strike"]) + (rfr + 0.5 * final_put_df["impliedVolatility"] ** 2) * final_put_df['TTM']) / (final_put_df["impliedVolatility"] * np.sqrt(final_put_df['TTM'])))
    final_put_df["Put Gamma"] = norm.pdf(final_put_df['d1']) / (current_price * final_put_df['impliedVolatility'] * np.sqrt(final_put_df['TTM']))

    return final_call_df, final_put_df

def calc_gex(final_call_df, final_put_df, current_price):
    
    '''
    GEX measures how dealers are exposed to gamma in the options market where gamma is the second derivative of price and the first derivative of delta (which is the first derivative of price).
    Dealers can be long (positive) or short (negative) gamma. Using GEX, we can see where dealers' risks are and predict their hedging activity.
        Long call: postive delta exposure. Short call: negative delta exposure.
        Long put: negative delta exposure. Short put: positive delta exposure.
        Short put / call: negative GEX
        Long put / call: positive GEX
    Gamma - lowest far OTM, far ITM, highest ATM
    Net short cumulative gamma - volatility expansion (dealers sell into dips, buy into rips) HOWEVER if in negative gamma but price goes up, dealers need to buy
    Net positive cumulative gamma - volatility compression (dealers buy into dips, sell at rips) HOWEVER if in positive gamma but price goes down, dealers need to sell
    '''

    final_call_df['GEX_call'] = final_call_df['Call Gamma'] * final_call_df['openInterest'] * 100 * current_price
    final_put_df['GEX_put'] = final_put_df['Put Gamma'] * final_put_df['openInterest'] * 100 * current_price * -1

    # Sum all GEX for each strike
    call_gamma_strike = final_call_df.groupby('strike', as_index = False)['GEX_call'].sum()
    put_gamma_strike = final_put_df.groupby('strike', as_index = False)['GEX_put'].sum()

    union_df_gamma_strike = pd.merge(call_gamma_strike, put_gamma_strike, on='strike', how='outer').fillna(0)
    union_df_gamma_strike['Total GEX'] = union_df_gamma_strike["GEX_call"] + union_df_gamma_strike["GEX_put"] 
    
    total_gamma = union_df_gamma_strike['Total GEX'].sum()

    return union_df_gamma_strike, final_call_df, final_put_df, total_gamma

def zero_gex_level(union_df_gamma_strike):

    '''
    The zero GEX level often serves as a pivot around which the market oscillates, with put support and call resistance acting as boundaries. 
    If the zero GEX level were outside this range say, above the call resistance or below the put support, it would imply a highly skewed options GEX distribution 
    (e.g., far more put GEX than call GEX where zero gamma is < PS and vice versa), which is less common in balanced markets.
    Even in skewed scenarios, the zero GEX level’s practical relevance diminishes if it lies beyond significant support or resistance, 
    as price rarely reaches extremes without breaking through the CR or PS first.
    '''

    idx_zero_gex = union_df_gamma_strike['Total GEX'].abs().idxmin()
    strike_zero_gex = union_df_gamma_strike.loc[idx_zero_gex, 'strike']
    return strike_zero_gex

def gamma_flip_zone(union_df_gamma_strike):
    
    union_df_gamma_strike['Sign'] = union_df_gamma_strike['Total GEX'].apply(lambda x: -1 if x <= 0 else 1)
    filter_negative_cum_gex = union_df_gamma_strike[union_df_gamma_strike['Sign'] == -1]

    if filter_negative_cum_gex.index[-1] == len(union_df_gamma_strike) - 1:
        idx_flip = filter_negative_cum_gex.index[-2]

    else:
        idx_flip = filter_negative_cum_gex.index[-1]

    return union_df_gamma_strike['strike'][idx_flip + 1]

# === EXPECTED MOVE ===
def get_expected_move_weekly(final_call_df, final_put_df, current_price):
    
    '''
    Only run Sat/Sun for weekly expected moves.
    '''

    if datetime.today().weekday() >= 5:
        # Weekly Expected Move
        exp_call = final_call_df["Expiration"][0]
        exp_put = final_call_df["Expiration"][0]

        truncated_call = final_call_df[final_call_df['Expiration'] == exp_call]
        truncated_put = final_put_df[final_put_df['Expiration'] == exp_put]

        last_itm_call_idx = truncated_call[truncated_call['inTheMoney']].index[-1]
        ask_px_itm_call = truncated_call['ask'][last_itm_call_idx]
        bid_px_itm_call = truncated_call['bid'][last_itm_call_idx]
        mid_px_itm_call = (bid_px_itm_call + ask_px_itm_call) / 2

        ask_px_itm_put = truncated_put['ask'][0]
        bid_px_itm_put = truncated_put['bid'][0]
        mid_px_itm_put = (bid_px_itm_put + ask_px_itm_put) / 2

        weekly_exp_move = ((ask_px_itm_call + ask_px_itm_put) + (mid_px_itm_call + mid_px_itm_put)) / 2
        weekly_expected_price = [round(current_price + float(weekly_exp_move), 2), round(current_price - float(weekly_exp_move), 2)]

        return weekly_expected_price
    
    else:
        print("Weekly move calculation can only be assessed after close on Friday.")
        return None

# === VISUALIZATION ===
def plot_CR_PS_ZG_PX_and_gex(ticker, data, CR, PS, ZG, union_df_gamma_strike, current_price, total_gamma, weekly_move):
   
    # Create a figure with 2 subplots (1 row, 2 columns)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot the stock prices and resistance/support/flip zones on the first subplot
    axes[0].plot(data['Close'], color='navy', label=f"{ticker} Stock Prices")
    axes[0].axhline(y=CR, color='r', linestyle='-.', label='Call Resistance')
    axes[0].axhline(y=PS, color='lime', linestyle='-.', label='Put Support')
    axes[0].axhline(y=ZG, color='b', linestyle='-.', label='Zero GEX Level')
    if weekly_move != None:
        axes[0].axhline(y=weekly_move[1], color='darkgreen', linestyle='--', label='Weekly Expected Move - Lower')
        axes[0].axhline(y=weekly_move[0], color='darkred', linestyle='--', label='Weekly Expected Move - Upper')

    axes[0].set_title(f"{ticker} Stock Prices", fontsize=14)
    axes[0].legend(fontsize=10, loc='lower right')
    axes[0].grid(True)
    axes[0].text(0, 0, f"Total Net Gex: {total_gamma}", fontsize = 12)

    # Sort the union_df_gamma_strike DataFrame
    union_df_gamma_strike = union_df_gamma_strike.sort_values(by='strike')
    strikes = union_df_gamma_strike['strike']
    total_gex = union_df_gamma_strike['Total GEX']

    # Plot Gamma Exposure (GEX) on the second subplot
    axes[1].bar(strikes, total_gex, color='blue', alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('Strike Price', fontsize=12)
    axes[1].set_ylabel('Total GEX', fontsize=12)
    axes[1].set_title(f'Gamma Exposure (GEX) by Strike \n Total Net Gex: {total_gamma}', fontsize=14)
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)
    axes[1].axvline(x=ZG, color='b', linestyle='-.', label='Zero GEX Level')
    axes[1].axvline(x=CR, color='r', linestyle='-.', label='Call Resistance')
    axes[1].axvline(x=current_price, color='cyan', linestyle='--', label='Current Price')
    axes[1].axvline(x=PS, color='g', linestyle='-.', label='Put Support')
    axes[1].legend(fontsize=10)

    # Rotate x-axis labels for readability
    axes[1].tick_params(axis='x', labelrotation=45, labelsize=10)

    # Adjust layout
    plt.tight_layout()

    # Show the plots + save
    plt.savefig(f"{ticker}.png")  # Save each plot separately
    plt.show()
    plt.close()

def main():
    
    try:
        watch_tickers = ['AAPL','UBER', 'META', 'MSFT', 'NFLX', 'GOOG', 'TSLA', 'NVDA', 'PLTR', 'MSTR', 'WMT', 'COST', 'ABBV', 'ARM', 'PYPL', 'RDDT', 'LRCX', 'MCD', 'CRWD', 'ARM']
        curr_watch_tickers = ['AAPL', 'MCD', 'GOOG', 'TSLA', 'NVDA', 'UBER', 'META', 'MSFT', 'PLTR', 'MSTR', 'RDDT', 'COIN']
        traded_tickers = ['AMZN', 'LLY', 'AVGO', 'COST']

        for ticker in curr_watch_tickers:
            data = yf.download(ticker, start = "2024-09-01")
            current_price = get_stock_data(ticker)[0]
            df = get_stock_data(ticker)[1]
            print("Ticker: ", ticker)

            final_call_df = call_option_data(df, current_price)
            CR = max_OI(final_call_df)
            print("Call Resistance: ", CR)
            
            final_put_df = put_option_data(df, current_price)
            PS = max_OI_put(final_put_df)
            print("Put Support: ", PS)
            
            final_call_df = days_exp(final_call_df, final_put_df)[0]
            final_put_df = days_exp(final_call_df, final_put_df)[1]
            final_call_df = calc_gamma(final_call_df, final_put_df, current_price, rfr = RISK_FREE_RATE)[0]
            final_put_df = calc_gamma(final_call_df, final_put_df, current_price, rfr = RISK_FREE_RATE)[1]
            union_df_gamma_strike = calc_gex(final_call_df, final_put_df, current_price)[0]
            final_call_df = calc_gex(final_call_df, final_put_df, current_price)[1]
            final_put_df = calc_gex(final_call_df, final_put_df, current_price)[2]
            total_gamma = calc_gex(final_call_df, final_put_df, current_price)[3]
            print("Total Cum. Net GEX:", total_gamma)
            zero_gamma = zero_gex_level(union_df_gamma_strike)
            print("Zero GEX Level: ", zero_gamma)
            GF = gamma_flip_zone(union_df_gamma_strike)
            print("Gamma Flip Zone: ", GF)

            weekly_move = get_expected_move_weekly(final_call_df, final_put_df, current_price)
            if weekly_move != None:
                print(f"Weekly Expected Move: {weekly_move}")

            plot_CR_PS_ZG_PX_and_gex(ticker, data, CR, PS, zero_gamma, union_df_gamma_strike, current_price, total_gamma, weekly_move)
    
    except:
        ticker = input("Type ticker to analyze: ").upper()
        data = yf.download(ticker, start = "2024-09-01")

        current_price = get_stock_data(ticker)[0]
        df = get_stock_data(ticker)[1]
        print("Ticker: ", ticker)

        final_call_df = call_option_data(df, current_price)
        CR = max_OI(final_call_df)
        print("Call Resistance: ", CR)
        
        final_put_df = put_option_data(df, current_price)
        PS = max_OI_put(final_put_df)
        print("Put Support: ", PS)
        
        final_call_df = days_exp(final_call_df, final_put_df)[0]
        final_put_df = days_exp(final_call_df, final_put_df)[1]
        final_call_df = calc_gamma(final_call_df, final_put_df, current_price, rfr = RISK_FREE_RATE)[0]
        final_put_df = calc_gamma(final_call_df, final_put_df, current_price, rfr = RISK_FREE_RATE)[1]
        union_df_gamma_strike = calc_gex(final_call_df, final_put_df, current_price)[0]
        final_call_df = calc_gex(final_call_df, final_put_df, current_price)[1]
        final_put_df = calc_gex(final_call_df, final_put_df, current_price)[2]
        total_gamma = calc_gex(final_call_df, final_put_df, current_price)[3]
        print("Total Cum. Net GEX:", total_gamma)
        zero_gamma = zero_gex_level(union_df_gamma_strike)
        print("Zero GEX Level: ", zero_gamma)
        GF = gamma_flip_zone(union_df_gamma_strike)
        print("Gamma Flip Zone: ", GF)

        weekly_move = get_expected_move_weekly(final_call_df, final_put_df, current_price)
        if weekly_move != None:
            print(f"Weekly Expected Move: {weekly_move}")

        plot_CR_PS_ZG_PX_and_gex(ticker, data, CR, PS, zero_gamma, union_df_gamma_strike, current_price, total_gamma, weekly_move)

main()