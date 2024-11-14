import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

#ticker = input("Type ticker to analyze: ").upper()

ticker = 'MSTR'

def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    current_price = stock.info['currentPrice']
    options_data = []
    options_expirations = stock.options
    
    for _ in options_expirations[0:4]:
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
        itm_calls = call_option_per_exp.loc[(call_option_per_exp['strike'] <= price_13_pct_itm) & (call_option_per_exp['inTheMoney'] == True)]
        otm_calls = call_option_per_exp.loc[(call_option_per_exp['strike'] > price_13_pct_itm) & (call_option_per_exp['inTheMoney'] == False)]
        combined_df = pd.concat([itm_calls, otm_calls], ignore_index = True)
        final_call_df = pd.concat([final_call_df, combined_df], ignore_index = True)

    final_call_df["Expiration"] = final_call_df['contractSymbol'].apply(lambda symbol: ''.join(char for char in symbol if char in '0123456789')[:6])
    final_call_df["Expiration"] = final_call_df['Expiration'].apply(lambda exp: '/'.join(exp[i:i + 2] for i in range(0, len(exp), 2)))

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
        otm_puts = row.loc[(row['strike'] > price_13_pct_otm) & (row['inTheMoney'] == False)]
        combined_df = pd.concat([itm_puts, otm_puts], ignore_index=True)
        final_put_df = pd.concat([final_put_df, combined_df], ignore_index=True)
    
    final_put_df["Expiration"] = final_put_df['contractSymbol'].apply(lambda symbol: ''.join(char for char in symbol if char in '0123456789')[:6])
    final_put_df["Expiration"] = final_put_df['Expiration'].apply(lambda exp: '/'.join(exp[i:i + 2] for i in range(0, len(exp), 2)))

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
def gamma():
    pass

def main():
    data = yf.download(ticker, start = "2024-11-01", end = "2024-12-01")
    data['Close'].plot()
    plt.title(f"{ticker} Stock Prices")

    current_price = get_stock_data(ticker)[0]
    df = get_stock_data(ticker)[1]
    final_call_df = call_option_data(df, current_price)
    print("Ticker: ", ticker)
    print("Call resistance: ", max_OI(final_call_df))
    plt.axhline(y = max_OI(final_call_df), color = 'r', linestyle = '-.') 
    final_put_df = put_option_data(df, current_price)
    print("Put support: ", max_OI_put(final_put_df))
    plt.axhline(y = max_OI_put(final_put_df), color = 'g', linestyle = '-.')

    plt.show()

main()
