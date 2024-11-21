import yfinance as yf
from py_vollib.black_scholes.greeks.analytical import gamma
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    current_price = stock.info['currentPrice']
    options_data = []
    options_expirations = stock.options
    
    for _ in options_expirations[1:6]:
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
    price_13_pct_otm = current_price * 1.13
    price_13_pct_itm = current_price * .87
    
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

    final_call_df['Call Gamma'] = final_call_df.apply(lambda row: gamma('c', current_price, row["strike"], row["TTM"], rfr, row["impliedVolatility"]), axis = 1)
    final_put_df['Put Gamma'] = final_call_df.apply(lambda row: gamma('p', current_price, row["strike"], row["TTM"], rfr, row["impliedVolatility"]), axis = 1)
    
    return final_call_df, final_put_df

def calc_gex(final_call_df, final_put_df):
    final_call_df['GEX_call'] = (final_call_df['Call Gamma'] * final_call_df['openInterest'])
    final_put_df['GEX_put'] = (final_put_df['Put Gamma'] * final_put_df['openInterest'])

    max_c_gamma = final_call_df.groupby('Expiration', as_index = False)['GEX_call'].max()
    gamma_lst = []

    for gamma in max_c_gamma['GEX_call']:
        gamma_lst.append(gamma)
    
    highest_gamma_call = final_call_df[final_call_df['GEX_call'].isin(gamma_lst)]
    strike_call = [strike for strike in highest_gamma_call['strike']]

    max_p_gamma = final_put_df.groupby('Expiration', as_index = False)['GEX_put'].max()
    gamma_lst = []

    for gamma in max_p_gamma['GEX_put']:
        gamma_lst.append(gamma)

    highest_gamma_put = final_put_df[final_put_df['GEX_put'].isin(gamma_lst)]
    strike_put = [strike for strike in highest_gamma_put['strike']]

    return sorted(list(set(strike_call))), sorted(list(set(strike_put)))
    
def main():
    ticker = input("Type ticker to analyze: ").upper()
    data = yf.download(ticker, start = "2024-11-01", end = "2024-12-01")
    data['Close'].plot()
    plt.title(f"{ticker} Stock Prices")

    current_price = get_stock_data(ticker)[0]
    df = get_stock_data(ticker)[1]
    final_call_df = call_option_data(df, current_price)
    print("Ticker: ", ticker)
    print("Call resistance: ", max_OI(final_call_df))
    plt.axhline(y = max_OI(final_call_df), color = 'r', linestyle = '-.', label = 'Call Resistance') 
    final_put_df = put_option_data(df, current_price)
    print("Put support: ", max_OI_put(final_put_df))
    plt.axhline(y = max_OI_put(final_put_df), color = 'g', linestyle = '-.', label = 'Put Support')
    plt.legend(fontsize=10)

    final_call_df = days_exp(final_call_df, final_put_df)[0]
    final_put_df = days_exp(final_call_df, final_put_df)[1]

    final_call_df = calc_gamma(final_call_df, final_put_df, current_price, rfr = 0.04404)[0]
    final_put_df = calc_gamma(final_call_df, final_put_df, current_price, rfr = 0.04404)[1]

    print('Call GEX', calc_gex(final_call_df, final_put_df)[0])
    print('Put GEX', calc_gex(final_call_df, final_put_df)[1])

    plt.show()

main()
