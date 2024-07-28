#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
import torch
from sklearn.metrics import mean_squared_error, r2_score

model = XGBRegressor(objective="reg:squarederror", random_state=13, tree_method = "hist",n_estimators = 2000,  colsample_bynode =0.45, learning_rate=0.05, max_depth=10, min_child_weight=1, subsample=0.5)

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
df_list = [df_eem,df_qqq,df_tlt]
df_final = df_gld.copy()
for df in df_list:
    df_final = df_final.merge(df,on='Date')
    
df_final.drop('Date',axis=1,inplace=True)


#%% Test with GLD only
def make_preds(df,comm,index):    
    df = df.drop('Date',axis=1)
    for i in range(60):
        lag_time =  str(i+1) + " day lag"
        df[lag_time] = df.loc[:,comm].shift((i+1))
        
    df.dropna(inplace=True)
    df.reset_index(inplace=True,drop=True)

    df_train = df.iloc[:index,:]
    df_test = df.iloc[index:,:]
    X_train = df_train.drop(comm,axis=1)
    X_test = df_test.drop(comm,axis=1)
    y_train = df_train.loc[:,comm]
    y_test = df_test.loc[:,comm]
    
    evalset = [(X_train,y_train),(X_test,y_test)]
        # Model
    model.fit(X_train,y_train,  verbose = False)
    X_test.reset_index(inplace=True, drop=True)
    for i, row in X_test.iterrows():
        if i == 60:
            break
        if i == 0:
            preds = []
            pred = model.predict(row.values.reshape(1,-1))
            preds.append(pred)
        else:
            
            for j, num in enumerate(preds[-1:-1]):
                row.iloc[j] = num
            pred = model.predict(row.values.reshape(1,-1))
            preds.append(pred)
    return preds




#%%

#Check columns vs lag
df_list = {'GLD':df_gld, 'EEM':df_eem,'QQQ':df_qqq,'TLT':df_tlt}
switches ={'GLD':0, 'EEM':0,'QQQ':0,'TLT':0}
portfolio = {'GLD':0, 'EEM':0,'QQQ':0,'TLT':0, 'Cash':10000}
buy_price = {'GLD':0, 'EEM':0,'QQQ':0,'TLT':0}
sell_price =  {'GLD':0, 'EEM':0,'QQQ':0,'TLT':0} 
turned_on = {'GLD':False, 'EEM':False,'QQQ':False,'TLT':False}
total_inv = []
for index in np.arange(len(df_gld)): 
    if (index >= 364)  and ((index-364)%30==0)and(df_final.shape[0]-index > 60):
        total = 0
        
        predictions = {}
        for comm in df_list:
            
            #Prediction
            predict = make_preds(df_list[comm],comm,index)
            predictions[comm] = predict
        for stock in df_list:
            turned_on[stock] = False
            if (df_list[stock].iloc[index].values[-1] > predictions[stock][-1][0]) and (switches[stock]!=0):
                print(f'turning off switch for {stock}')
                switches[stock] = 0
                sell_price[stock] = df_list[stock].iloc[index].values[-1]
                if buy_price[stock]!= 0:
                    sale_ratio = sell_price[stock]/buy_price[stock]
                    portfolio['Cash']+=sale_ratio*portfolio[stock]
                    print(f'Selling {stock}, sell price: ${sell_price[stock]}, buy price: {buy_price[stock]} ratio: {sale_ratio}, total: {sale_ratio*portfolio[stock]}')
                    portfolio[stock] = 0
            elif df_list[stock].iloc[index].values[-1]<= predictions[stock][-1][0]:
                 if switches[stock] !=1:
                    switches[stock] = 1
                    total+=1
                    turned_on[stock] = True
                    buy_price[stock] = df_list[stock].iloc[index].values[-1]
                
       
        
        if total != 0 and portfolio['Cash'] != 0:
            ratio = 1/total
            cash = portfolio['Cash']
            print(f'{index}Investing {ratio*cash} into {total} accounts')
            
            for stock in turned_on:
                if turned_on[stock]:
                    portfolio[stock] += ratio*portfolio['Cash']
                    
            portfolio['Cash'] = 0
            
            running_total = []
            for stock in df_list:
                if buy_price[stock]!=0:
                    sale_ratio = sell_price[stock]/buy_price[stock]
                    running_total.append(sale_ratio*portfolio[stock])
            current = sum(running_total)+portfolio['Cash']
        total_inv.append(current)
#%%     Issue: Often buying stock at different buy ins, need a way to keep a running total of stock value.   
import matplotlib.pyplot as plt    
            
plt.plot(total_inv)        
        
            
        
   
# %%
import numpy as np
total_return = (sum(portfolio.values())-10000)/10000

annual_return = np.power(1+total_return,1/19) -1
print(annual_return)

# %%
sp500 = pd.read_csv(r"C:\Users\jeffu\Downloads\SPY.csv")
prices = sp500['Open']

tot = (prices.iloc[-1]-prices.iloc[0])/prices.iloc[0]
ann = np.power(1+tot,1/19)-1
ann




