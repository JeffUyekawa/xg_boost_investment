#%%
import pandas as pd

df_gld = pd.read_csv(r"C:\Users\jeffu\Downloads\GLD (1).csv")
df_eem = pd.read_csv(r"C:\Users\jeffu\Downloads\EEM (1).csv")
df_qqq = pd.read_csv(r"C:\Users\jeffu\Downloads\QQQ (1).csv")
df_tlt = pd.read_csv(r"C:\Users\jeffu\Downloads\TLT (1).csv")

df_gld.rename(columns={'Open':'GLD'},inplace=True)
df_eem.rename(columns={'Open':'EEM'},inplace=True)
df_qqq.rename(columns={'Open':"QQQ"},inplace=True)
df_tlt.rename(columns={'Open':'TLT'},inplace=True)

df_gld = df_gld.iloc[:,:2]
df_eem = df_eem.iloc[:,:2]
df_qqq = df_qqq.iloc[:,:2]
df_tlt = df_tlt.iloc[:,:2]
#%%
df_list = [df_eem,df_qqq,df_tlt]
df_final = df_gld.copy()
for df in df_list:
    df_final = df_final.merge(df,on='Date')
# %%
df_final['GLD_Lag'] = df_final['GLD'].shift(54)
df_final['EEM_Lag'] = df_final['EEM'].shift(54)
df_final['QQQ_Lag'] = df_final['QQQ'].shift(54)
df_final['TLT_Lag'] = df_final['TLT'].shift(54)
# %%
df = df_final.dropna()
# %%
df_final.to_csv(r"C:\Users\jeffu\Downloads\investment_dataset.csv")

#%%
## Buy only when currently uninvested 
#Check columns vs lag
checks = {'GLD':'GLD_Lag', 'EEM':'EEM_Lag','QQQ':'QQQ_Lag','TLT':'TLT_Lag'}
switches ={'GLD':0, 'EEM':0,'QQQ':0,'TLT':0}
portfolio = {'GLD':0, 'EEM':0,'QQQ':0,'TLT':0, 'Cash':10000}
buy_price = {'GLD':0, 'EEM':0,'QQQ':0,'TLT':0}
sell_price =  {'GLD':0, 'EEM':0,'QQQ':0,'TLT':0} 
turned_on = {'GLD':False, 'EEM':False,'QQQ':False,'TLT':False}

total_inv = []
for index,row in df.iterrows():
    total = 0
    for stock in checks:
        turned_on[stock] = False
        if (row[stock]<=row[checks[stock]]) and (switches[stock]!=0):
            print(f'turning off switch for {stock}')
            switches[stock] = 0
            sell_price[stock] = row[stock]
            if buy_price[stock]!= 0:
                sale_ratio = sell_price[stock]/buy_price[stock]
                portfolio['Cash']+=sale_ratio*portfolio[stock]
                print(f'Selling {stock}, sell price: ${sell_price[stock]}, buy price: {buy_price[stock]} ratio: {sale_ratio}, total: {sale_ratio*portfolio[stock]}')
                portfolio[stock] = 0
        elif row[stock] > row[checks[stock]]:
            if switches[stock] !=1:
                switches[stock] = 1
                total+=1
                turned_on[stock] = True
                buy_price[stock] = row[stock]
                
    if total != 0 and portfolio['Cash'] != 0:
        ratio = 1/total
        cash = portfolio['Cash']
        print(f'{index}Investing {ratio*cash} into {total} accounts')
        
        for stock in turned_on:
            if turned_on[stock]:
                
                portfolio[stock] += ratio*portfolio['Cash']
                
                
        portfolio['Cash'] = 0
        
        running_total = []
        for stock in checks:
            if buy_price[stock]!=0:
                sale_ratio = sell_price[stock]/buy_price[stock]
                running_total.append(sale_ratio*portfolio[stock])
        current = sum(running_total)+portfolio['Cash']
    total_inv.append(current)
    
import matplotlib.pyplot as plt    
            
plt.plot(total_inv)        
   
# %%
import numpy as np
total_return = (total_inv[-1]-10000)/10000

annual_return = np.power(1+total_return,1/19) -1
print(annual_return)

# %%
sp500 = pd.read_csv(r"C:\Users\jeffu\Downloads\SPY.csv")

# %%
sp500
# %%
prices = sp500['Open']

tot = (prices.iloc[-1]-prices.iloc[0])/prices.iloc[0]
ann = np.power(1+tot,1/19)-1
ann
# %%

#%%
## Immediately buy if still higher than lag
#Check columns vs lag
checks = {'GLD':'GLD_Lag', 'EEM':'EEM_Lag','QQQ':'QQQ_Lag','TLT':'TLT_Lag'}
switches ={'GLD':0, 'EEM':0,'QQQ':0,'TLT':0}
portfolio = {'GLD':0, 'EEM':0,'QQQ':0,'TLT':0, 'Cash':10000}
buy_price = {'GLD':0, 'EEM':0,'QQQ':0,'TLT':0}
sell_price =  {'GLD':0, 'EEM':0,'QQQ':0,'TLT':0} 
turned_on = {'GLD':False, 'EEM':False,'QQQ':False,'TLT':False}

total_inv = []
for index,row in df.iterrows():
    
    for stock in checks:
        turned_on[stock] = False
        if (row[stock]<=row[checks[stock]]) and (switches[stock]!=0):
            print(f'turning off switch for {stock}')
            switches[stock] = 0
            sell_price[stock] = row[stock]
            if buy_price[stock]!= 0:
                sale_ratio = sell_price[stock]/buy_price[stock]
                portfolio['Cash']+=sale_ratio*portfolio[stock]
                print(f'Selling {stock}, sell price: ${sell_price[stock]}, buy price: {buy_price[stock]} ratio: {sale_ratio}, total: {sale_ratio*portfolio[stock]}')
                portfolio[stock] = 0
        elif row[stock] > row[checks[stock]]:
            if switches[stock] !=1:
                switches[stock] = 1
                
                turned_on[stock] = True
                buy_price[stock] = row[stock]
    total = sum(switches.values())          
    if total != 0 and portfolio['Cash'] != 0:
        ratio = 1/total
        cash = portfolio['Cash']
        print(f'{index}Investing {ratio*cash} into {total} accounts')
        
        for stock in switches:
            if switches[stock]==1:
                portfolio[stock] = row[stock]/buy_price[stock]*portfolio[stock]
                buy_price[stock]= row[stock]
                portfolio[stock] += ratio*portfolio['Cash']
                
                
        portfolio['Cash'] = 0
        
        running_total = []
        for stock in checks:
            if buy_price[stock]!=0:
                sale_ratio = sell_price[stock]/buy_price[stock]
                running_total.append(sale_ratio*portfolio[stock])
        current = sum(running_total)+portfolio['Cash']
    total_inv.append(current)
    
import matplotlib.pyplot as plt    
            
plt.plot(total_inv)        
   
# %%
import numpy as np
total_return = (total_inv[-1]-10000)/10000

annual_return = np.power(1+total_return,1/19) -1
print(annual_return)

# %%
