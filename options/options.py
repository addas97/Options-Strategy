import yfinance as yf
import pandas as pd

#ticker = input("Type ticker to analyze: ").upper()

ticker = 'NVDA'

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
def call_option_data(df):
    call_df = df[["Exp", "Calls"]]
    final_call_df = pd.DataFrame()
    
    for call_option_per_exp in call_df["Calls"]:
        last_4_itm_call = call_option_per_exp.loc[call_option_per_exp['inTheMoney'] == True][-1:-5:-1]
        all_otm_call = call_option_per_exp.loc[call_option_per_exp['inTheMoney'] == False][:]
        final_call_df = final_call_df._append([last_4_itm_call, all_otm_call], ignore_index = True)

    final_call_df["Expiration"] = final_call_df['contractSymbol'].apply(lambda symbol: ''.join(char for char in symbol if char in '0123456789')[:6])
    final_call_df["Expiration"] = final_call_df['Expiration'].apply(lambda exp: '/'.join(exp[i:i + 2] for i in range(0, len(exp), 2)))

    return final_call_df

def max_OI(final_call_df):
    max_OI = final_call_df.groupby("Expiration", as_index = False)['openInterest'].max()
    oi_lst = []
    
    for i in max_OI['openInterest']:
        oi_lst.append(i)
    
    highest_OI = final_call_df[final_call_df['openInterest'].isin(oi_lst)]

    strikes = []
    for i in highest_OI['strike']:
        strikes.append(i)
    
    return max(list(set(strikes)))

# === PUT SUPPORT LEVEL ===
def put_option_data(df):
    put_data = df[["Exp", "Puts"]]
    final_put_df = pd.DataFrame()

    for row in put_data['Puts']:
        itm_puts = row.loc[row['inTheMoney'] == True][-1:-5:-1]
        otm_puts = row.loc[row['inTheMoney'] == False][:]
        final_put_df = final_put_df._append([itm_puts, otm_puts], ignore_index = False)
    
    final_put_df["Expiration"] = final_put_df['contractSymbol'].apply(lambda symbol: ''.join(char for char in symbol if char in '0123456789')[:6])
    final_put_df["Expiration"] = final_put_df['Expiration'].apply(lambda exp: '/'.join(exp[i:i + 2] for i in range(0, len(exp), 2)))

    return final_put_df

def max_OI_put(final_put_df):
    max_OI = final_put_df.groupby('Expiration', as_index = False)['openInterest'].max()
    oi_lst = []

    for oi in max_OI['openInterest']:
        oi_lst.append(oi)

    highest_put_oi = final_put_df[final_put_df['openInterest'].isin(oi_lst)]

    strikes = []
    for strike in highest_put_oi['strike']:
        strikes.append(strike)
    
    return max(list(set(strikes)))

# === GAMMA LEVELS ===
def gamma():
    pass

def main():
    current_price = get_stock_data(ticker)[0]
    df = get_stock_data(ticker)[1]
    final_call_df = call_option_data(df)
    print("Call resistance: ", max_OI(final_call_df))
    final_put_df = put_option_data(df)
    print("Put support: ", max_OI_put(final_put_df))

main()