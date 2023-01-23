#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys
import pandas as pd
from django.contrib.staticfiles.storage import staticfiles_storage
import django
import pickle
from keras.models import load_model
import numpy as np 
import yfinance as yf
from nsepy import get_history
from statsmodels.tsa.seasonal import seasonal_decompose
from datetime import date
import datetime
import talib
from scipy.stats import norm
from yahoofinancials import YahooFinancials
from nsepython import *

def get_data(symbol,start,end):
    try:
        df = yf.download(symbol+'.NS',start=str(start),end=str(end))
        df['Symbol'] = symbol
        df = df[['Symbol','Open','High','Low','Close','Adj Close','Volume']]
        df.to_csv(staticfiles_storage.path(symbol+'.csv'))
    except:
        try:
            df = get_history(symbol,start=start,end=end)
            df['Symbol'] = symbol
            df.reset_index(inplace=True)
            df = df[['Date','Symbol','Open','High','Low','Close','Last','Volume']]
            df = df.rename({'Last':'Adj Close'},axis=1)
            df.to_csv(staticfiles_storage.path(symbol+'.csv'),index=False)
        except:
            print(f'Data not found for {symbol}')


def get_macro_data(symbol,start,end):
    try:
        df = get_history(symbol,start=start,end=end,index=True)
        df['Symbol'] = symbol
        df.reset_index(inplace=True)
        df = df[['Date','Symbol','Open','High','Low','Close','Volume']]
        df['Adj Close'] = df['Close']
        df = df[['Date','Symbol','Open','High','Low','Close','Adj Close','Volume']]
        df.to_csv(staticfiles_storage.path(symbol+'.csv'),index=False)   
    except Exception as e:
        print(e)



def create_macro_vars(index):
    df = pd.read_csv(staticfiles_storage.path(index+'.csv'),header=0)
    df = df.sort_values(by='Date')
    df = df[['Date','Adj Close']]
    df['returns'] = np.log(df['Adj Close']/df['Adj Close'].shift(1))
    for j in range(5,30,5):
        df[f'MA{j}'] = df['Adj Close'].rolling(window=j).mean()
        df[f'MA{j}_ratio'] = np.log(df['Adj Close']/df[f'MA{j}'])
        df[f'High{j}'] = df['Adj Close'].rolling(window=j).max()
        df[f'Low{j}'] = df['Adj Close'].rolling(window=j).min()
        df[f'High{j}_ratio'] = np.log(df['Adj Close']/df[f'High{j}'])
        df[f'Low{j}_ratio'] = np.log(df[f'Low{j}']/df['Adj Close'])
        df[f'Voltality{j}'] = df['returns'].rolling(window=j).std()
    for i in range(1,26):
        df[f'log_return_{i}'] = np.log(df['Adj Close'].shift(i-1)/df['Adj Close'].shift(i))
    df.dropna(inplace=True)
    df.to_csv(staticfiles_storage.path(f'Variables/{index}.csv'),index=False)

def pred_macro_vars(index,last_trade_date):
    df = pd.read_csv(staticfiles_storage.path(index+'.csv'),header=0)
    df = df.sort_values(by='Date')
    df = df[['Date','Adj Close']]
    df['returns'] = np.log(df['Adj Close']/df['Adj Close'].shift(1))
    for j in range(5,30,5):
        df[f'MA{j}'] = df['Adj Close'].rolling(window=j).mean()
        df[f'MA{j}_ratio'] = np.log(df['Adj Close']/df[f'MA{j}'])
        df[f'High{j}'] = df['Adj Close'].rolling(window=j).max()
        df[f'Low{j}'] = df['Adj Close'].rolling(window=j).min()
        df[f'High{j}_ratio'] = np.log(df['Adj Close']/df[f'High{j}'])
        df[f'Low{j}_ratio'] = np.log(df[f'Low{j}']/df['Adj Close'])
        df[f'Voltality{j}'] = df['returns'].rolling(window=j).std()
    for i in range(1,26):
        df[f'log_return_{i}'] = np.log(df['Adj Close'].shift(i-1)/df['Adj Close'].shift(i))
    df = df[df['Date']==last_trade_date]
    df.to_csv(staticfiles_storage.path(f'Pred_Vars/{index}.csv'),index=False)

def create_variables(symbol):
    df = pd.read_csv(staticfiles_storage.path(symbol+'.csv'),header=0)
    df = df.sort_values(by='Date')
    df = df[(df['Volume']!=0) & (df['Volume'].notnull())]
    df['future_return'] = np.log(df['Adj Close'].shift(-1)/df['Adj Close'])
    df['lift'] = df['future_return'].apply(lambda x: 1 if x >0 else 0)
    df['returns'] = np.log(df['Adj Close']/df['Adj Close'].shift(1))
    for i in range(1,26):
        df[f'log_return_{i}'] = np.log(df['Adj Close'].shift(i-1)/df['Adj Close'].shift(i))
        df[f'Open_Chg{i}'] = np.log(df['Open'].shift(i-1)/df['Open'].shift(i))
        df[f'High_Chg{i}'] = np.log(df['High'].shift(i-1)/df['High'].shift(i))
        df[f'Low_Chg{i}'] = np.log(df['Low'].shift(i-1)/df['Low'].shift(i))
        df[f'Range_Chg{i}'] = np.log((df['High'].shift(i-1)-df['Low'].shift(i-1))/(df['High'].shift(i)-df['Low'].shift(i)))
        df[f'Volume_Chg{i}'] = np.log(df['Volume'].shift(i-1)/df['Volume'].shift(i))
    for j in range(5,30,5):
        df[f'Voltality{j}'] = df['returns'].rolling(window=j).std()
        df[f'MA{j}'] = df['Adj Close'].rolling(window=j).mean()
        df[f'High{j}'] = df['Adj Close'].rolling(window=j).max()
        df[f'Low{j}'] = df['Adj Close'].rolling(window=j).min()
        df[f'MA{j}_ratio'] = np.log(df['Adj Close']/df[f'MA{j}'])
        df[f'High{j}_ratio'] = np.log(df['Adj Close']/df[f'High{j}'])
        df[f'Low{j}_ratio'] = np.log(df[f'Low{j}']/df['Adj Close'])
        df[f'Price_Voltality{j}'] = df['Adj Close'].rolling(window=j).std()
        df[f'Bollinger{j}_ratio'] = (df['Adj Close']-df[f'MA{j}']+2*df[f'Price_Voltality{j}'])/(4*df[f'Price_Voltality{j}'])
        df[f'ADX{j}'] = talib.ADX(df['High'],df['Low'],df['Close'],j)/100
        df[f'RSI{j}'] = talib.RSI(df['Adj Close'],j)/100
    df.dropna(inplace=True)
    df.to_csv(staticfiles_storage.path(f'Variables/{symbol}.csv'),index=False)


def pred_variables(symbol,last_trade_date):
    df = pd.read_csv(staticfiles_storage.path(symbol+'.csv'),header=0)
    df = df.sort_values(by='Date')
    df = df[(df['Volume']!=0) & (df['Volume'].notnull())]
    df['future_return'] = np.log(df['Adj Close'].shift(-1)/df['Adj Close'])
    df['lift'] = df['future_return'].apply(lambda x: 1 if x >0 else 0)
    df['returns'] = np.log(df['Adj Close']/df['Adj Close'].shift(1))
    for i in range(1,26):
        df[f'log_return_{i}'] = np.log(df['Adj Close'].shift(i-1)/df['Adj Close'].shift(i))
        df[f'Open_Chg{i}'] = np.log(df['Open'].shift(i-1)/df['Open'].shift(i))
        df[f'High_Chg{i}'] = np.log(df['High'].shift(i-1)/df['High'].shift(i))
        df[f'Low_Chg{i}'] = np.log(df['Low'].shift(i-1)/df['Low'].shift(i))
        df[f'Range_Chg{i}'] = np.log((df['High'].shift(i-1)-df['Low'].shift(i-1))/(df['High'].shift(i)-df['Low'].shift(i)))
        df[f'Volume_Chg{i}'] = np.log(df['Volume'].shift(i-1)/df['Volume'].shift(i))
    for j in range(5,30,5):
        df[f'Voltality{j}'] = df['returns'].rolling(window=j).std()
        df[f'MA{j}'] = df['Adj Close'].rolling(window=j).mean()
        df[f'High{j}'] = df['Adj Close'].rolling(window=j).max()
        df[f'Low{j}'] = df['Adj Close'].rolling(window=j).min()
        df[f'MA{j}_ratio'] = np.log(df['Adj Close']/df[f'MA{j}'])
        df[f'High{j}_ratio'] = np.log(df['Adj Close']/df[f'High{j}'])
        df[f'Low{j}_ratio'] = np.log(df[f'Low{j}']/df['Adj Close'])
        df[f'Price_Voltality{j}'] = df['Adj Close'].rolling(window=j).std()
        df[f'Bollinger{j}_ratio'] = (df['Adj Close']-df[f'MA{j}']+2*df[f'Price_Voltality{j}'])/(4*df[f'Price_Voltality{j}'])
        df[f'ADX{j}'] = talib.ADX(df['High'],df['Low'],df['Close'],j)/100
        df[f'RSI{j}'] = talib.RSI(df['Adj Close'],j)/100
    df = df[df['Date']==last_trade_date]
    df.to_csv(staticfiles_storage.path(f'Pred_Vars/{symbol}.csv'),index=False)




def create_dev_data(symbol):
    df = pd.read_csv(staticfiles_storage.path(f'Variables/{symbol}.csv'),header=0)
    portfolio_shares = pd.read_excel(staticfiles_storage.path('portfolio shares.xlsx'),header=0)
    sector_symbol = portfolio_shares.loc[portfolio_shares['Symbol']==f'{symbol}','Sector_Symbol'].values[0]
    sector_data = pd.read_csv(staticfiles_storage.path(f'Variables/{sector_symbol}.csv'),header=0)
    cap_symbol = portfolio_shares.loc[portfolio_shares['Symbol']==f'{symbol}','Cap_Symbol'].values[0]
    cap_data = pd.read_csv(staticfiles_storage.path(f'Variables/{cap_symbol}.csv'),header=0)
    nifty500 = pd.read_csv(staticfiles_storage.path('Variables/NIFTY 500.csv'),header=0)
    gs813 = pd.read_csv(staticfiles_storage.path('Variables/NIFTY GS 8 13YR.csv'),header=0)
    gs10 = pd.read_csv(staticfiles_storage.path('Variables/NIFTY GS 10YR.csv'),header=0)
    gs10cln = pd.read_csv(staticfiles_storage.path('Variables/NIFTY GS 10YR CLN.csv'),header=0)
    gs48 = pd.read_csv(staticfiles_storage.path('Variables/NIFTY GS 4 8YR.csv'),header=0)
    gs1115 = pd.read_csv(staticfiles_storage.path('Variables/NIFTY GS 11 15YR.csv'),header=0)
    gs15plus = pd.read_csv(staticfiles_storage.path('Variables/NIFTY GS 15YRPLUS.csv'),header=0)
    gscompsite = pd.read_csv(staticfiles_storage.path('Variables/NIFTY GS COMPSITE.csv'),header=0)
    df1 = df.merge(sector_data,how='inner',on='Date',suffixes=('','_sector'))
    df2 = df1.merge(nifty500,how='inner',on='Date',suffixes=('','_nifty500'))
    df3 = df2.merge(gs813,how='inner',on='Date',suffixes=('','_gs813'))
    df4 = df3.merge(gs10,how='inner',on='Date',suffixes=('','_gs10'))
    df5 = df4.merge(gs10cln,how='inner',on='Date',suffixes=("","_gs10cln"))
    df6 = df5.merge(gs48,how="inner",on="Date",suffixes=("","_gs48"))
    df7 = df6.merge(gs1115,how="inner",on="Date",suffixes=("","_gs1115"))
    df8 = df7.merge(gs15plus,how="inner",on="Date",suffixes=("","_gs15plus"))
    df9 = df8.merge(gscompsite,how="inner",on="Date",suffixes=("","_gscompsite"))
    df10 = df9.merge(cap_data,how='inner',on='Date',suffixes=('','_cap'))
    
    return df10

def create_pred_data(symbol):
    df = pd.read_csv(staticfiles_storage.path(f'Pred_Vars/{symbol}.csv'),header=0)
    portfolio_shares = pd.read_excel(staticfiles_storage.path('portfolio shares.xlsx'),header=0)
    sector_symbol = portfolio_shares.loc[portfolio_shares['Symbol']==f'{symbol}','Sector_Symbol'].values[0]
    sector_data = pd.read_csv(staticfiles_storage.path(f'Pred_Vars/{sector_symbol}.csv'),header=0)
    cap_symbol = portfolio_shares.loc[portfolio_shares['Symbol']==f'{symbol}','Cap_Symbol'].values[0]
    cap_data = pd.read_csv(staticfiles_storage.path(f'Pred_Vars/{cap_symbol}.csv'),header=0)
    nifty500 = pd.read_csv(staticfiles_storage.path('Pred_Vars/NIFTY 500.csv'),header=0)
    gs813 = pd.read_csv(staticfiles_storage.path('Pred_Vars/NIFTY GS 8 13YR.csv'),header=0)
    gs10 = pd.read_csv(staticfiles_storage.path('Pred_Vars/NIFTY GS 10YR.csv'),header=0)
    gs10cln = pd.read_csv(staticfiles_storage.path('Pred_Vars/NIFTY GS 10YR CLN.csv'),header=0)
    gs48 = pd.read_csv(staticfiles_storage.path('Pred_Vars/NIFTY GS 4 8YR.csv'),header=0)
    gs1115 = pd.read_csv(staticfiles_storage.path('Pred_Vars/NIFTY GS 11 15YR.csv'),header=0)
    gs15plus = pd.read_csv(staticfiles_storage.path('Pred_Vars/NIFTY GS 15YRPLUS.csv'),header=0)
    gscompsite = pd.read_csv(staticfiles_storage.path('Pred_Vars/NIFTY GS COMPSITE.csv'),header=0)
    df1 = df.merge(sector_data,how='inner',on='Date',suffixes=('','_sector'))
    df2 = df1.merge(nifty500,how='inner',on='Date',suffixes=('','_nifty500'))
    df3 = df2.merge(gs813,how='inner',on='Date',suffixes=('','_gs813'))
    df4 = df3.merge(gs10,how='inner',on='Date',suffixes=('','_gs10'))
    df5 = df4.merge(gs10cln,how='inner',on='Date',suffixes=("","_gs10cln"))
    df6 = df5.merge(gs48,how="inner",on="Date",suffixes=("","_gs48"))
    df7 = df6.merge(gs1115,how="inner",on="Date",suffixes=("","_gs1115"))
    df8 = df7.merge(gs15plus,how="inner",on="Date",suffixes=("","_gs15plus"))
    df9 = df8.merge(gscompsite,how="inner",on="Date",suffixes=("","_gscompsite"))
    df10 = df9.merge(cap_data,how='inner',on='Date',suffixes=('','_cap'))
    
    return df10



def get_equation(model):
    n_layers = len(model.layers)
    mat1 = np.matrix(model.layers[1].get_weights()[0])
    inputs = mat1.shape[0]
    mat2 = np.matrix(model.layers[1].get_weights()[1])
    mat3 = np.concatenate((mat1,mat2),axis=0)
    for i in range(2,n_layers):
        mat4 = np.matrix(model.layers[i].get_weights()[0])
        mat5 = np.matmul(mat3,mat4)
        mat5[inputs,:]+=np.matrix(model.layers[i].get_weights()[1])
        
        mat3 = mat5
    return [float(x) for x in mat5[:,0]]


def main():
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Trading.settings')
    django.setup()
    from Broker.models import Stock,Option,Valuation 
    portfolio_shares = pd.read_excel(staticfiles_storage.path('portfolio shares.xlsx'),header=0)
    pca_components = pd.read_excel(staticfiles_storage.path('pca_components.xlsx'),header=0)
    pca_components.rename({'Unnamed: 0':'PC'},axis=1,inplace=True)
    input_cols = pca_components.drop('PC',axis=1).columns
    pca = pickle.load(open(staticfiles_storage.path('pca'),'rb'))
    return_model = load_model(staticfiles_storage.path('return_model.h5'))
    eqn_data = pd.read_excel(staticfiles_storage.path('dynamic_eqn.xlsx'),header=0)
    eqn_data.sort_values(by='Date',inplace=True)
    macros = pd.read_excel(staticfiles_storage.path('macro indexes.xlsx'),header=0)
    end = date.today()
    start_opt = end - datetime.timedelta(days=5)
    last_train_date = list(eqn_data['Date'])[-1]
    last_train = datetime.date(year=int(last_train_date[:4]),month=int(last_train_date[5:7]),day=int(last_train_date[8:]))
    dates_pending = []
    for i in range((end - last_train).days+1):
        dates_pending.append(last_train+datetime.timedelta(days=i))
    start = dates_pending[0] - datetime.timedelta(days=100)
    get_macro_data('NIFTY 500',start=start,end=end)
    niftydata = pd.read_csv(staticfiles_storage.path('NIFTY 500.csv'),header=0)
    niftydata.sort_values(by='Date',inplace=True)
    last_trade_date =  sorted(list(niftydata['Date']))[-1]
    last_trade = datetime.date(year=int(last_trade_date[:4]),month=int(last_trade_date[5:7]),day=int(last_trade_date[8:]))
    create_macro_vars('NIFTY 500')
    df_check = pd.read_csv(staticfiles_storage.path('Variables/NIFTY 500.csv'),header=0)
    df_check.sort_values(by='Date',inplace=True)
    df_check['future_return'] = np.log(df_check['Adj Close'].shift(-1)/df_check['Adj Close'])
    df_check.dropna(inplace=True)
    df_check.sort_values(by='Date',inplace=True)
    last_available_date = list(df_check['Date'])[-1]
    last_available = datetime.date(year=int(last_available_date[:4]),month=int(last_available_date[5:7]),day=int(last_available_date[8:]))
    dates_pending = []
    for i in range(1,(last_available - last_train).days+1):
        dates_pending.append(last_train+datetime.timedelta(days=i))
        
    if last_available_date == last_train_date:
        try:
            pred_data = pd.read_csv(staticfiles_storage.path('pred_data.csv'),header=0)
            for col in input_cols:
                pred_data[col] = pred_data[col].apply(lambda x: max(-1,x) if x <0 else min(x,1))
            
            dates = sorted(list(pred_data['Date'].unique()))
            if dates[-1] == last_trade_date:
                pred_data['expected_return'] = return_model.predict(pca.transform(pred_data[input_cols]))
                pred_data['expected price'] = pred_data['Close']*np.exp(pred_data['expected_return'])
                pred_data.to_csv(staticfiles_storage.path('pred_data.csv'),index=False)
                stock_predictions = pd.DataFrame()
                stock_predictions['Symbol'] = pred_data['Symbol']
                stock_predictions['EOD Price'] = pred_data['Close']
                stock_predictions['Expected Price'] = pred_data['expected price']
                stock_predictions['Company'] = stock_predictions['Symbol'].apply(lambda x: 
                    portfolio_shares.loc[portfolio_shares['Symbol']==x,'Bank/Company Name'].values[0])
                stock_predictions['Sector'] = stock_predictions['Symbol'].apply(lambda x: 
                    portfolio_shares.loc[portfolio_shares['Symbol']==x,'Sector'].values[0])
                stock_predictions = stock_predictions[['Sector','Company','Symbol','EOD Price','Expected Price']]
                stock_predictions.to_excel(staticfiles_storage.path('stock predictions.xlsx'),index=False)
            else:
                for symbol in list(portfolio_shares['Symbol']):
                    get_data(symbol,start=start,end=end)
                for index in list(macros['Index']):
                    get_macro_data(index,start=start,end=end)
                for sector in list(portfolio_shares['Sector_Symbol'].unique()):
                    get_macro_data(sector,start=start,end=end)
                for cap in list(portfolio_shares['Cap_Symbol'].unique()):
                    get_macro_data(cap,start=start,end=end)

                for symbol in list(portfolio_shares['Symbol']):
                    pred_variables(symbol,last_trade_date=last_trade_date)

                for index in list(macros['Index']):
                    pred_macro_vars(index,last_trade_date=last_trade_date)

                for sector in list(portfolio_shares['Sector_Symbol'].unique()):
                    pred_macro_vars(sector,last_trade_date=last_trade_date)
                for cap in list(portfolio_shares['Cap_Symbol'].unique()):
                    pred_macro_vars(cap,last_trade_date=last_trade_date)



                symbols = list(portfolio_shares['Symbol'])
                pred_data = create_pred_data(symbols[0])
                for symbol in symbols[1:]:
                    pred_data = pred_data.append(create_pred_data(symbol))
                for col in input_cols:
                    pred_data[col] = pred_data[col].apply(lambda x: max(-1,x) if x <0 else min(x,1))
                pred_data['expected_return'] = return_model.predict(pca.transform(pred_data[input_cols]))
                pred_data['expected price'] = pred_data['Close']*np.exp(pred_data['expected_return'])
                pred_data.to_csv(staticfiles_storage.path('pred_data.csv'),index=False)
                stock_predictions = pd.DataFrame()
                stock_predictions['Symbol'] = pred_data['Symbol']
                stock_predictions['EOD Price'] = pred_data['Close']
                stock_predictions['Expected Price'] = pred_data['expected price']
                stock_predictions['Company'] = stock_predictions['Symbol'].apply(lambda x: 
                    portfolio_shares.loc[portfolio_shares['Symbol']==x,'Bank/Company Name'].values[0])
                stock_predictions['Sector'] = stock_predictions['Symbol'].apply(lambda x: 
                    portfolio_shares.loc[portfolio_shares['Symbol']==x,'Sector'].values[0])
                stock_predictions = stock_predictions[['Sector','Company','Symbol','EOD Price','Expected Price']]
                stock_predictions.to_excel(staticfiles_storage.path('stock predictions.xlsx'),index=False)

        except:
            for symbol in list(portfolio_shares['Symbol']):
                    get_data(symbol,start=start,end=end)
            for index in list(macros['Index']):
                get_macro_data(index,start=start,end=end)
            for sector in list(portfolio_shares['Sector_Symbol'].unique()):
                get_macro_data(sector,start=start,end=end)
            for cap in list(portfolio_shares['Cap_Symbol'].unique()):
                get_macro_data(cap,start=start,end=end)

            for symbol in list(portfolio_shares['Symbol']):
                pred_variables(symbol,last_trade_date=last_trade_date)

            for index in list(macros['Index']):
                pred_macro_vars(index,last_trade_date=last_trade_date)

            for sector in list(portfolio_shares['Sector_Symbol'].unique()):
                pred_macro_vars(sector,last_trade_date=last_trade_date)
            for cap in list(portfolio_shares['Cap_Symbol'].unique()):
                pred_macro_vars(cap,last_trade_date=last_trade_date)

            symbols = list(portfolio_shares['Symbol'])
            pred_data = create_pred_data(symbols[0])
            for symbol in symbols[1:]:
                pred_data = pred_data.append(create_pred_data(symbol))
            for col in input_cols:
                pred_data[col] = pred_data[col].apply(lambda x: max(-1,x) if x <0 else min(x,1))
            pred_data['expected_return'] = return_model.predict(pca.transform(pred_data[input_cols]))
            pred_data['expected price'] = pred_data['Close']*np.exp(pred_data['expected_return'])
            pred_data.to_csv(staticfiles_storage.path('pred_data.csv'),index=False)
            stock_predictions = pd.DataFrame()
            stock_prediction['Symbol'] = pred_data['Symbol']
            stock_predictions['EOD Price'] = pred_data['Close']
            stock_predictions['Expected Price'] = pred_data['expected price']
            stock_predictions['Company'] = stock_predictions['Symbol'].apply(lambda x: 
                portfolio_shares.loc[portfolio_shares['Symbol']==x,'Bank/Company Name'].values[0])
            stock_predictions['Sector'] = stock_predictions['Symbol'].apply(lambda x: 
                portfolio_shares.loc[portfolio_shares['Symbol']==x,'Sector'].values[0])
            stock_predictions = stock_predictions[['Sector','Company','Symbol','EOD Price','Expected Price']]
            stock_predictions.to_excel(staticfiles_storage.path('stock predictions.xlsx'),index=False)

    else:
        for symbol in list(portfolio_shares['Symbol']):
            get_data(symbol,start=start,end=end)
        for index in list(macros['Index']):
            get_macro_data(index,start=start,end=end)
        for sector in list(portfolio_shares['Sector_Symbol'].unique()):
            get_macro_data(sector,start=start,end=end)
        for cap in list(portfolio_shares['Cap_Symbol'].unique()):
            get_macro_data(cap,start=start,end=end)

        for symbol in list(portfolio_shares['Symbol']):
            create_variables(symbol)

        for index in list(macros['Index']):
            create_macro_vars(index)

        for sector in list(portfolio_shares['Sector_Symbol'].unique()):
            create_macro_vars(sector)

        for cap in list(portfolio_shares['Cap_Symbol'].unique()):
            create_macro_vars(cap)



        for symbol in list(portfolio_shares['Symbol']):
            pred_variables(symbol,last_trade_date=last_trade_date)

        for index in list(macros['Index']):
            pred_macro_vars(index,last_trade_date=last_trade_date)

        for sector in list(portfolio_shares['Sector_Symbol'].unique()):
            pred_macro_vars(sector,last_trade_date=last_trade_date)

        for cap in list(portfolio_shares['Cap_Symbol'].unique()):
            pred_macro_vars(cap,last_trade_date=last_trade_date)

        symbols = list(portfolio_shares['Symbol'])
        dev_data = create_dev_data(symbols[0])
        for symbol in symbols[1:]:
            dev_data = dev_data.append(create_dev_data(symbol))
        for col in input_cols:
            dev_data[col] = dev_data[col].apply(lambda x: max(-1,x) if x <0 else min(x,1))
        dev_data = dev_data[dev_data['Date'].isin([str(datep) for datep in dates_pending])]
        dev_data.to_csv(staticfiles_storage.path('dev_data.csv'),index=False)
        dates = sorted(dev_data['Date'].unique())
        dev_data['expected_return'] = np.nan
        eq_model = pd.read_excel(staticfiles_storage.path('dynamic_eqn.xlsx'),header=0)
        eq_model.sort_values(by='Date',inplace=True)
        for i in range(len(dates)):
            pca_components2 = pca_components.copy()
            try:
                dev_data.loc[dev_data['Date']==dates[i],'expected_return'] = return_model.predict(pca.transform(dev_data.loc[dev_data['Date']==dates[i],input_cols]))
                return_model.fit(pca.transform(dev_data.loc[dev_data['Date']==dates[i],input_cols]),dev_data.loc[dev_data['Date']==dates[i],'future_return'],epochs=100)
                eqn = get_equation(return_model)
                intercept = eqn[-1]
                means = pca.mean_
                for col in input_cols:
                    pca_components2[col] = pca_components2[col]*np.array(eqn[:-1])
                final_eqn = pca_components2[input_cols].sum(axis=0)
                intercept -= np.sum(np.multiply(final_eqn,means))
                eq_model = eq_model.append(pd.DataFrame([[dates[i]]+list(final_eqn.values)+[intercept]],columns=eq_model.columns))
            except Exception as e:
                pass
        return_model.save(staticfiles_storage.path('return_model.h5'))
        eq_model.to_excel(staticfiles_storage.path('dynamic_eqn.xlsx'),index=False)

        history = pd.read_excel(staticfiles_storage.path('historical_data.xlsx'),header=0)
        if dev_data.shape[0] > 0:
            dev_data['Actual Price'] = dev_data['Adj Close']
            dev_data['Predicted Price'] = dev_data['Adj Close']*np.exp(dev_data['expected_return'])
            dev_data['return observed'] = dev_data['future_return']
            dev_data['return predicted'] = dev_data['expected_return']
            dev_data['lift observed'] = dev_data['return observed'].apply(lambda x: 1 if x>0 else 0)
            dev_data['lift predicted'] = dev_data['return predicted'].apply(lambda x: 1 if x>0 else 0)
            dev_data['correct prediction'] = dev_data[['lift observed','lift predicted']].apply(lambda x: 1 if x[0]==x[1] else 0,axis=1)
            
            dev_data = dev_data[list(history.columns)]
            dev_data.sort_values(by='Date',inplace=True)
            for symbol in symbols:
                last_price = list(dev_data.loc[dev_data['Symbol']==symbol,'Actual Price'])[-1]
                dev_data = dev_data.append(pd.DataFrame([[symbol,last_trade_date,last_price,np.nan,
                    np.nan,np.nan,np.nan,np.nan,np.nan]],columns=list(history.columns)))

            for symbol in list(dev_data['Symbol'].unique()):
                append_data = dev_data[dev_data['Symbol']==symbol]
                append_data.sort_values(by='Date',inplace=True)
                append_data['Predicted Price'] = append_data['Predicted Price'].shift(1)
                append_data['return observed'] = append_data['return observed'].shift(1)
                append_data['return predicted'] = append_data['return predicted'].shift(1)
                append_data['lift observed'] = append_data['lift observed'].shift(1)
                append_data['lift predicted'] = append_data['lift predicted'].shift(1)
                append_data['correct prediction'] = append_data['correct prediction'].shift(1)
                append_data.dropna(inplace=True)
                append_data = append_data[list(history.columns)]
                history = history.append(append_data)


            history.to_excel(staticfiles_storage.path('historical_data.xlsx'),index=False)

        pred_data = create_pred_data(symbols[0])
        for symbol in symbols[1:]:
            pred_data = pred_data.append(create_pred_data(symbol))
        for col in input_cols:
            pred_data[col] = pred_data[col].apply(lambda x: max(-1,x) if x <0 else min(x,1))
        if pred_data.shape[0] > 0:
            pred_data['expected_return'] = return_model.predict(pca.transform(pred_data[input_cols]))
            pred_data['expected price'] = pred_data['Close']*np.exp(pred_data['expected_return'])
            pred_data.to_csv(staticfiles_storage.path('pred_data.csv'),index=False)
            stock_predictions = pd.DataFrame()
            stock_predictions['Symbol'] = pred_data['Symbol']
            stock_predictions['EOD Price'] = pred_data['Close']
            stock_predictions['Expected Price'] = pred_data['expected price']
            stock_predictions['Company'] = stock_predictions['Symbol'].apply(lambda x: 
                portfolio_shares.loc[portfolio_shares['Symbol']==x,'Bank/Company Name'].values[0])
            stock_predictions['Sector'] = stock_predictions['Symbol'].apply(lambda x: 
                portfolio_shares.loc[portfolio_shares['Symbol']==x,'Sector'].values[0])
            stock_predictions = stock_predictions[['Sector','Company','Symbol','EOD Price','Expected Price']]
            stock_predictions.to_excel(staticfiles_storage.path('stock predictions.xlsx'),index=False)


    df = pd.read_excel(staticfiles_storage.path('stock predictions.xlsx'),header=0)
    for symbol in list(df['Symbol']):
        if Stock.objects.filter(Symbol=symbol).exists():
            stock = Stock.objects.get(Symbol=symbol)
            stock.EOD_Price = np.round(df.loc[df['Symbol']==symbol,'EOD Price'].values[0],2)
            stock.Expected_Price = np.round(df.loc[df['Symbol']==symbol,'Expected Price'].values[0],2)
            stock.save()
        else:
            sector = df.loc[df['Symbol']==symbol,'Sector'].values[0]
            company = df.loc[df['Symbol']==symbol,'Company'].values[0]
            EOD_Price = np.round(df.loc[df['Symbol']==symbol,'EOD Price'].values[0],2)
            Expected_Price = np.round(df.loc[df['Symbol']==symbol,'Expected Price'].values[0],2)

            stock = Stock(Sector=sector,Company=company,Symbol=symbol,EOD_Price=EOD_Price,Expected_Price=Expected_Price)
            stock.save()

    eqn_data = pd.read_excel(staticfiles_storage.path('dynamic_eqn.xlsx'),header=0)
    eqn_data.sort_values(by='Date',inplace=True)
    latest_eqn = eqn_data.iloc[-1,:][1:]
    pred_data = pd.read_csv(staticfiles_storage.path('pred_data.csv'),header=0)
    input_cols = latest_eqn.index[:-1]
    intercept = latest_eqn[-1]
    labels = ['_nifty500','_gs813','_gs10',"_gs10cln","_gs48","_gs1115","_gs15plus","_gscompsite"]
    macro_specific_cols = []
    for label in labels:
        macro_specific_cols += [col for col in input_cols if label in str.lower(col)]

    sector_specific_cols = [col for col in input_cols if '_sector' in str.lower(col)]

    cap_specific_cols = [col for col in input_cols if '_cap' in str.lower(col)]

    comp_specific_cols = list(set(input_cols)-set(macro_specific_cols)-set(sector_specific_cols)-set(cap_specific_cols))
    reason_codes = pd.DataFrame(columns=['Symbol','Macro','Sector','Capital','Company','Market'])
    for symbol in list(pred_data['Symbol'].unique()):
        macro_cont = np.sum(np.multiply(np.array(pred_data[pred_data['Symbol']==symbol][macro_specific_cols]),np.array(latest_eqn[macro_specific_cols])))
        sector_cont = np.sum(np.multiply(np.array(pred_data[pred_data['Symbol']==symbol][sector_specific_cols]),np.array(latest_eqn[sector_specific_cols])))
        cap_cont = np.sum(np.multiply(np.array(pred_data[pred_data['Symbol']==symbol][cap_specific_cols]),np.array(latest_eqn[cap_specific_cols])))
        comp_cont = np.sum(np.multiply(np.array(pred_data[pred_data['Symbol']==symbol][comp_specific_cols]),np.array(latest_eqn[comp_specific_cols])))
        reason_codes = reason_codes.append(pd.DataFrame([[symbol,macro_cont,sector_cont,cap_cont,comp_cont,intercept]],columns=reason_codes.columns))

    reason_codes['Macro'] = np.exp(reason_codes['Macro'])
    reason_codes['Sector'] = np.exp(reason_codes['Sector'])
    reason_codes['Capital'] = np.exp(reason_codes['Capital'])
    reason_codes['Company'] = np.exp(reason_codes['Company'])
    reason_codes['Market'] = np.exp(reason_codes['Market'])

    reason_codes['Net Impact'] = reason_codes['Macro']*reason_codes['Sector']*reason_codes['Capital']*reason_codes['Company']*reason_codes['Market']
    reason_codes['inc_dec'] = reason_codes['Net Impact']-1
    reason_codes.to_excel(staticfiles_storage.path('reason_codes.xlsx'),index=False)

    for stock in Stock.objects.all():
        try:
            symbol = stock.Symbol
            x = YahooFinancials(f'{symbol}.NS')
            req_dict = x.get_financial_stmts('annual', 'income')['incomeStatementHistory'][f'{symbol}.NS']
            df = pd.DataFrame()
            for i in req_dict[-1::-1]:
                col = list(i.keys())[0]
                df_append = pd.DataFrame(pd.Series(i[col],index=i[col].keys()),columns=[col])
                df = pd.concat([df,df_append],axis=1)
            df.fillna(0,inplace=True)
            df.to_excel(staticfiles_storage.path(f'Income/{symbol}.xlsx'))

            req_dict = x.get_financial_stmts('annual', 'balance')['balanceSheetHistory'][f'{symbol}.NS']
            df = pd.DataFrame()
            for i in req_dict[-1::-1]:
                col = list(i.keys())[0]
                df_append = pd.DataFrame(pd.Series(i[col],index=i[col].keys()),columns=[col])
                df = pd.concat([df,df_append],axis=1)
            df.fillna(0,inplace=True)
            df.to_excel(staticfiles_storage.path(f'Balance/{symbol}.xlsx'))

        except:
            pass

    for stock in Stock.objects.all():
        try:
            if Valuation.objects.filter(Stock=stock).exists():
                valuation = Valuation.objects.get(Stock=stock)
            else:
                valuation = Valuation()
                valuation.Stock = stock 

            symbol = stock.Symbol 
            ticker = yf.Ticker(f'{symbol}.NS')
            shareholding_breakup = ticker.major_holders[[1,0]]
            shareholding_breakup.columns = ['Holder','Percent']
            shareholding_breakup.to_excel(staticfiles_storage.path(f'Holding_Breakup/{symbol}.xlsx'),index=False)
            est_shares = int(portfolio_shares.loc[portfolio_shares['Symbol']==symbol,'Shares'].values[0])
            valuation.Shares_Outstanding = est_shares
            income = pd.read_excel(staticfiles_storage.path(f'Income/{symbol}.xlsx'),header=0)
            balance_sheet = pd.read_excel(staticfiles_storage.path(f'Balance/{symbol}.xlsx'),header=0)
            income.rename({'Unnamed: 0':'Component'},axis=1,inplace=True)
            balance_sheet.rename({'Unnamed: 0':'Component'},axis=1,inplace=True)
            income.set_index('Component',inplace=True)
            balance_sheet.set_index('Component',inplace=True)
            income = income.T
            balance_sheet = balance_sheet.T
            income_cols = ['totalRevenue','costOfRevenue','grossProfit','researchDevelopment','sellingGeneralAdministrative',
                            'otherOperatingExpenses','totalOperatingExpenses','ebit','interestExpense',
                            'totalOtherIncomeExpenseNet','incomeBeforeTax','incomeTaxExpense','netIncomeFromContinuingOps']
            for col in input_cols:
                try:
                    print(income[col])
                except:
                    income[col]=0
            income = income[income_cols]

            income['Depreciation/Amortization'] = income['totalOperatingExpenses']-income['costOfRevenue']-income['researchDevelopment']-income['sellingGeneralAdministrative']-income['otherOperatingExpenses']
            income['InterestIncome'] = income['totalOtherIncomeExpenseNet']-income['interestExpense']
            income = income[['totalRevenue','costOfRevenue','grossProfit','researchDevelopment','sellingGeneralAdministrative',
                 'otherOperatingExpenses','Depreciation/Amortization','totalOperatingExpenses','ebit','interestExpense','InterestIncome',
                'totalOtherIncomeExpenseNet','incomeBeforeTax','incomeTaxExpense','netIncomeFromContinuingOps']]
            balance_cols = ['cash','shortTermInvestments','netReceivables','inventory','otherCurrentAssets',
                               'totalCurrentAssets','propertyPlantEquipment','longTermInvestments','goodWill',
                              'intangibleAssets','otherAssets','capitalSurplus','commonStock','retainedEarnings',
                              'otherStockholderEquity','minorityInterest','accountsPayable','otherCurrentLiab',
                              'totalCurrentLiabilities','otherLiab','longTermDebt']
            for col in balance_cols:
                try:
                    print(balance_sheet[col])
                except:
                    balance_sheet[col]=0
            balance_sheet = balance_sheet[balance_cols]
            balance_sheet['Securities'] = balance_sheet['totalCurrentAssets'] - balance_sheet['cash'] - balance_sheet['shortTermInvestments']-balance_sheet['netReceivables']-balance_sheet['inventory']-balance_sheet['otherCurrentAssets']
            balance_sheet['totalAssets'] = balance_sheet['totalCurrentAssets']+balance_sheet['propertyPlantEquipment']+balance_sheet['longTermInvestments']+balance_sheet['goodWill']+balance_sheet['intangibleAssets']+balance_sheet['otherAssets']
            balance_sheet['totalStockholderEquity'] = balance_sheet['capitalSurplus']+balance_sheet['commonStock']+balance_sheet['retainedEarnings']+balance_sheet['otherStockholderEquity']
            balance_sheet['totalEquity'] = balance_sheet['totalStockholderEquity'] + balance_sheet['minorityInterest']
            balance_sheet['shortTermBorrowings'] = balance_sheet['totalCurrentLiabilities'] - balance_sheet['accountsPayable'] - balance_sheet['otherCurrentLiab']
            balance_sheet['totalLiab'] = balance_sheet['totalCurrentLiabilities'] + balance_sheet['otherLiab'] + balance_sheet['longTermDebt']
            balance_sheet['Total Capital'] = balance_sheet['totalLiab'] + balance_sheet['totalEquity']
            balance_sheet['Net Working Capital'] = balance_sheet['totalCurrentAssets'] - balance_sheet['totalCurrentLiabilities']
            balance_sheet = balance_sheet[['cash','shortTermInvestments','netReceivables','inventory','otherCurrentAssets','Securities',
                               'totalCurrentAssets','propertyPlantEquipment','longTermInvestments','goodWill',
                              'intangibleAssets','otherAssets','totalAssets','capitalSurplus','commonStock','retainedEarnings',
                              'otherStockholderEquity','totalStockholderEquity','minorityInterest','totalEquity',
                               'accountsPayable','otherCurrentLiab','shortTermBorrowings','totalCurrentLiabilities',
                               'otherLiab','longTermDebt','totalLiab','Total Capital','Net Working Capital']]
            projections = income.join(balance_sheet)
            projections.index = [int(x[:4]) for x in projections.index]
            projections.sort_index(axis=0,inplace=True)
            last_year = projections.index[-1]
            for i in range(last_year+1,last_year+6):
                projections.loc[i,:] = [np.nan for j in range(len(projections.columns))]

            indexes = [('Actual',x) if x <= last_year else ('Projection',x) for x in projections.index]
            index = pd.MultiIndex.from_tuples(indexes, names=["value", "year"])
            projections.index = index
            first_year = projections.index[0][1]
            revenue_growth_rate = (projections.loc[('Actual',last_year),'totalRevenue']/projections.loc[('Actual',first_year),'totalRevenue'])**(1/(last_year-first_year))-1
            last_revenue = projections.loc[('Actual',last_year),'totalRevenue']
            for i in range(1,6):
                projections.loc[('Projection',last_year+i),'totalRevenue'] = last_revenue*(1+revenue_growth_rate)**i

            cost_of_revenue_per_sale = projections.loc[('Actual',last_year),'costOfRevenue']/projections.loc[('Actual',last_year),'totalRevenue']
            for i in range(1,6):
                projections.loc[('Projection',last_year+i),'costOfRevenue'] = cost_of_revenue_per_sale*projections.loc[('Projection',last_year+i),'totalRevenue']

            for i in range(1,6):
                projections.loc[('Projection',last_year+i),'grossProfit'] = projections.loc[('Projection',last_year+i),'totalRevenue'] - projections.loc[('Projection',last_year+i),'costOfRevenue']

            research_rate = projections.loc[('Actual',last_year),'researchDevelopment']/projections.loc[('Actual',last_year),'totalRevenue']
            for i in range(1,6):
                projections.loc[('Projection',last_year+i),'researchDevelopment'] = research_rate*projections.loc[('Projection',last_year+i),'totalRevenue']
            gen_ad_rate = projections.loc[('Actual',last_year),'sellingGeneralAdministrative']/projections.loc[('Actual',last_year),'totalRevenue']
            for i in range(1,6):
                projections.loc[('Projection',last_year+i),'sellingGeneralAdministrative'] = gen_ad_rate*projections.loc[('Projection',last_year+i),'totalRevenue']
            oth_opt_rev = projections.loc[('Actual',last_year),'otherOperatingExpenses']/projections.loc[('Actual',last_year),'totalRevenue']
            for i in range(1,6):
                projections.loc[('Projection',last_year+i),'otherOperatingExpenses'] = oth_opt_rev*projections.loc[('Projection',last_year+i),'totalRevenue']
            fixed_asset_turnover_ratio = projections.loc[('Actual',last_year),'propertyPlantEquipment']/projections.loc[('Actual',last_year),'totalRevenue']
            for i in range(1,6):
                projections.loc[('Projection',last_year+i),'propertyPlantEquipment'] = fixed_asset_turnover_ratio*projections.loc[('Projection',last_year+i),'totalRevenue']
            depreciation_rate = 1/(1+ projections.loc[('Actual',last_year),'propertyPlantEquipment']/projections.loc[('Actual',last_year),'Depreciation/Amortization'])
            for i in range(1,6):
                projections.loc[('Projection',last_year+i),'Depreciation/Amortization'] = depreciation_rate*(1-depreciation_rate)*projections.loc[('Projection',last_year+i),'propertyPlantEquipment']
            for i in range(1,6):
                projections.loc[('Projection',last_year+i),'totalOperatingExpenses'] = projections.loc[('Projection',last_year+i),'costOfRevenue']+projections.loc[('Projection',last_year+i),'researchDevelopment']+projections.loc[('Projection',last_year+i),'sellingGeneralAdministrative']+projections.loc[('Projection',last_year+i),'otherOperatingExpenses']+projections.loc[('Projection',last_year+i),'Depreciation/Amortization']
            for i in range(1,6):
                projections.loc[('Projection',last_year+i),'ebit'] = projections.loc[('Projection',last_year+i),'totalRevenue']-projections.loc[('Projection',last_year+i),'totalOperatingExpenses']
            interest_rate = projections.loc[('Actual',last_year),'interestExpense']/(projections.loc[('Actual',last_year),'shortTermBorrowings']+
                                                                        projections.loc[('Actual',last_year),'otherLiab']
                                                                        +projections.loc[('Actual',last_year),'longTermDebt'])
            net_receivables_per_revenue = projections.loc[('Actual',last_year),'netReceivables']/projections.loc[('Actual',last_year),'totalRevenue']
            inventory_per_revenue = projections.loc[('Actual',last_year),'inventory']/projections.loc[('Actual',last_year),'totalRevenue']
            acct_payable_per_revenue = projections.loc[('Actual',last_year),'accountsPayable']/projections.loc[('Actual',last_year),'totalRevenue']
            net_wc_per_revenue = projections.loc[('Actual',last_year),'Net Working Capital']/projections.loc[('Actual',last_year),'totalRevenue']
            for i in range(1,6):
                projections.loc[('Projection',last_year+i),'Net Working Capital'] = net_wc_per_revenue*projections.loc[('Projection',last_year+i),'totalRevenue']
            for i in range(1,6):
                projections.loc[('Projection',last_year+i),'netReceivables'] = net_receivables_per_revenue*projections.loc[('Projection',last_year+i),'totalRevenue']
            for i in range(1,6):
                projections.loc[('Projection',last_year+i),'inventory'] = inventory_per_revenue*projections.loc[('Projection',last_year+i),'totalRevenue']
            for i in range(1,6):
                projections.loc[('Projection',last_year+i),'accountsPayable'] = acct_payable_per_revenue*projections.loc[('Projection',last_year+i),'totalRevenue']
            for i in range(1,6):
                projections.loc[('Projection',last_year+i),'cash'] = projections.loc[('Actual',last_year),'cash']
            for i in range(1,6):
                projections.loc[('Projection',last_year+i),'shortTermInvestments'] = projections.loc[('Actual',last_year),'shortTermInvestments']
            for i in range(1,6):
                projections.loc[('Projection',last_year+i),'otherCurrentAssets'] = projections.loc[('Actual',last_year),'otherCurrentAssets']
            for i in range(1,6):
                projections.loc[('Projection',last_year+i),'Securities'] = projections.loc[('Actual',last_year),'Securities']
            for i in range(1,6):
                projections.loc[('Projection',last_year+i),'totalCurrentAssets'] = projections.loc[('Projection',last_year+i),'cash']+projections.loc[('Projection',last_year+i),'shortTermInvestments']+projections.loc[('Projection',last_year+i),'netReceivables']+projections.loc[('Projection',last_year+i),'inventory']+projections.loc[('Projection',last_year+i),'Securities']+projections.loc[('Projection',last_year+i),'otherCurrentAssets']
        
            for i in range(1,6):
                projections.loc[('Projection',last_year+i),'totalCurrentLiabilities'] = -projections.loc[('Projection',last_year+i),'Net Working Capital']+projections.loc[('Projection',last_year+i),'totalCurrentAssets'] 

            for i in range(1,6):
                projections.loc[('Projection',last_year+i),'otherCurrentLiab'] = projections.loc[('Actual',last_year),'otherCurrentLiab']

            for i in range(1,6):
                projections.loc[('Projection',last_year+i),'shortTermBorrowings'] = projections.loc[('Actual',last_year),'totalCurrentLiabilities']-projections.loc[('Projection',last_year+i),'otherCurrentLiab']-projections.loc[('Projection',last_year+i),'accountsPayable']

            for i in range(1,6):
                projections.loc[('Projection',last_year+i),'longTermInvestments'] = projections.loc[('Actual',last_year),'longTermInvestments']
        

            for i in range(1,6):
                projections.loc[('Projection',last_year+i),'goodWill'] = projections.loc[('Actual',last_year),'goodWill']

            for i in range(1,6):
                projections.loc[('Projection',last_year+i),'intangibleAssets'] = projections.loc[('Actual',last_year),'intangibleAssets']

            for i in range(1,6):
                projections.loc[('Projection',last_year+i),'otherAssets'] = projections.loc[('Actual',last_year),'otherAssets']

            for i in range(1,6):
                projections.loc[('Projection',last_year+i),'otherLiab'] = projections.loc[('Actual',last_year),'otherLiab']


            for i in range(1,6):
                projections.loc[('Projection',last_year+i),'longTermDebt'] = projections.loc[('Actual',last_year),'longTermDebt']


            for i in range(1,6):
                projections.loc[('Projection',last_year+i),'totalAssets'] = projections.loc[('Projection',last_year+i),'totalCurrentAssets']+projections.loc[('Projection',last_year+i),'propertyPlantEquipment']+projections.loc[('Projection',last_year+i),'longTermInvestments']+projections.loc[('Projection',last_year+i),'goodWill']+projections.loc[('Projection',last_year+i),'intangibleAssets']+projections.loc[('Projection',last_year+i),'otherAssets']


            for i in range(1,6):
                projections.loc[('Projection',last_year+i),'totalLiab'] = projections.loc[('Projection',last_year+i),'totalCurrentLiabilities']+projections.loc[('Projection',last_year+i),'otherLiab']+projections.loc[('Projection',last_year+i),'longTermDebt']


            for i in range(1,6):
                projections.loc[('Projection',last_year+i),'capitalSurplus'] = projections.loc[('Actual',last_year),'capitalSurplus']


            for i in range(1,6):
                projections.loc[('Projection',last_year+i),'commonStock'] = projections.loc[('Actual',last_year),'commonStock']


            for i in range(1,6):
                projections.loc[('Projection',last_year+i),'otherStockholderEquity'] = projections.loc[('Actual',last_year),'otherStockholderEquity']

            for i in range(1,6):
                projections.loc[('Projection',last_year+i),'minorityInterest'] = projections.loc[('Actual',last_year),'minorityInterest']


            for i in range(1,6):
                projections.loc[('Projection',last_year+i),'retainedEarnings'] = projections.loc[('Projection',last_year+i),'totalAssets']-projections.loc[('Projection',last_year+i),'totalLiab']-projections.loc[('Projection',last_year+i),'minorityInterest']-projections.loc[('Projection',last_year+i),'capitalSurplus']-projections.loc[('Projection',last_year+i),'commonStock']-projections.loc[('Projection',last_year+i),'otherStockholderEquity']


            for i in range(1,6):
                projections.loc[('Projection',last_year+i),'totalStockholderEquity'] = projections.loc[('Projection',last_year+i),'capitalSurplus']+projections.loc[('Projection',last_year+i),'commonStock']+projections.loc[('Projection',last_year+i),'otherStockholderEquity']+projections.loc[('Projection',last_year+i),'retainedEarnings']


            for i in range(1,6):
                projections.loc[('Projection',last_year+i),'totalEquity'] = projections.loc[('Projection',last_year+i),'totalStockholderEquity'] +projections.loc[('Projection',last_year+i),'minorityInterest'] 


            for i in range(1,6):
                projections.loc[('Projection',last_year+i),'Total Capital'] = projections.loc[('Projection',last_year+i),'totalEquity'] +projections.loc[('Projection',last_year+i),'totalLiab'] 


            interest_rate = projections.loc[('Actual',last_year),'interestExpense']/(projections.loc[('Actual',last_year),'shortTermBorrowings']+
                                                                        projections.loc[('Actual',last_year),'otherLiab']+
                                                                        projections.loc[('Actual',last_year),'longTermDebt'])

            tax_rate = projections.loc[('Actual',last_year),'incomeTaxExpense']/projections.loc[('Actual',last_year),'incomeBeforeTax']
            for i in range(1,6):
                projections.loc[('Projection',last_year+i),'interestExpense'] = interest_rate*(projections.loc[('Projection',last_year+i),'shortTermBorrowings']+
                                                                        projections.loc[('Projection',last_year+i),'otherLiab']+
                                                                        projections.loc[('Projection',last_year+i),'longTermDebt'])

            for i in range(1,6):
                projections.loc[('Projection',last_year+i),'InterestIncome'] = 0 
        

            for i in range(1,6):
                projections.loc[('Projection',last_year+i),'totalOtherIncomeExpenseNet'] = projections.loc[('Projection',last_year+i),'InterestIncome'] + projections.loc[('Projection',last_year+i),'interestExpense'] 



            for i in range(1,6):
                projections.loc[('Projection',last_year+i),'incomeBeforeTax'] = projections.loc[('Projection',last_year+i),'ebit'] + projections.loc[('Projection',last_year+i),'totalOtherIncomeExpenseNet']



            for i in range(1,6):
                projections.loc[('Projection',last_year+i),'incomeTaxExpense'] = projections.loc[('Projection',last_year+i),'incomeBeforeTax']*tax_rate



            for i in range(1,6):
                projections.loc[('Projection',last_year+i),'netIncomeFromContinuingOps'] = projections.loc[('Projection',last_year+i),'incomeBeforeTax']*(1-tax_rate)


            projections = projections.T
            projections.to_excel(staticfiles_storage.path(f'Projections/{symbol}.xlsx'))
            pred_data = pd.read_csv(staticfiles_storage.path('pred_data.csv'),header=0)
            stock_returns = pred_data.loc[pred_data['Symbol']==symbol,[f'log_return_{i}' for i in range(1,26)]]
            market_returns = pred_data.loc[pred_data['Symbol']==symbol,[f'log_return_{i}_nifty500' for i in range(1,26)]]
            market_voltality = pred_data.loc[pred_data['Symbol']==symbol,'Voltality25_nifty500'].values[0]
            beta = np.cov(stock_returns,market_returns)[0,1]/(market_voltality**2)
            valuation.Beta = beta
            cost_of_equity = (1-beta)*3.5/100+beta*11.5/100
            valuation.Cost_Of_Equity = cost_of_equity
            cost_of_debt = interest_rate
            valuation.Cost_Of_Debt = cost_of_debt
            debt_to_equity = projections.loc['totalLiab',('Actual',last_year)]/projections.loc['totalEquity',('Actual',last_year)]
            valuation.Debt_to_Equity = debt_to_equity
            cost_of_capital = cost_of_equity*(1/(1+debt_to_equity))+cost_of_debt*(1/(1+1/debt_to_equity))
            valuation.Cost_Of_Capital = cost_of_capital
            discounting_factors = [1/(1+cost_of_capital)**i for i in range(1,6)]
            ebits = projections.loc['ebit',[('Projection',last_year+i) for i in range(1,6)]]
            nopats = ebits*(1-tax_rate)
            dep_and_amortize = projections.loc['Depreciation/Amortization',[('Projection',last_year+i) for i in range(1,6)]]
            wc_increase = [projections.loc['Net Working Capital',('Projection',last_year+1)]-projections.loc['Net Working Capital',('Actual',last_year)]]
            wc_increase += [projections.loc['Net Working Capital',('Projection',last_year+i)]-projections.loc['Net Working Capital',('Projection',last_year+i-1)] for i in range(2,6)]
            less_capex = [-projections.loc['propertyPlantEquipment',('Projection',last_year+1)]+projections.loc['propertyPlantEquipment',('Actual',last_year)]]
            less_capex += [-projections.loc['propertyPlantEquipment',('Projection',last_year+i)]+projections.loc['propertyPlantEquipment',('Projection',last_year+i-1)] for i in range(2,6)]
            debt_free_cash = nopats + dep_and_amortize + wc_increase + less_capex
            present_values = []
            terminal_values = []
            for i in range(5):
                present_values.append(np.dot(debt_free_cash[:i+1],discounting_factors[:i+1]))
                terminal_values.append(debt_free_cash[i]/cost_of_capital*discounting_factors[i])
            dcf_values = np.array(present_values)+np.array(terminal_values)
            investments = [projections.loc['longTermInvestments',('Actual',last_year)] for i in range(5)]
            debt = [projections.loc['shortTermBorrowings',('Actual',last_year)]+projections.loc['otherLiab',('Actual',last_year)]+projections.loc['longTermDebt',('Actual',last_year)] for i in range(5)]
            cash = [projections.loc['cash',('Actual',last_year)] for i in range(5)]
            minority_interest = [projections.loc['minorityInterest',('Actual',last_year)] for i in range(5)]
            dcf_equity_values = np.array(dcf_values)+np.array(investments)-np.array(debt)+np.array(cash)+np.array(minority_interest)
            dcf_per_share = dcf_equity_values/est_shares
            valuation.Value1 = np.round(dcf_per_share[0],2)
            valuation.Value2 = np.round(dcf_per_share[1],2)
            valuation.Value3 = np.round(dcf_per_share[2],2)
            valuation.Value4 = np.round(dcf_per_share[3],2)
            valuation.Value5 = np.round(dcf_per_share[4],2)
            price = stock.EOD_Price
            EV = est_shares*price+debt[0]-cash[0]+minority_interest[0]
            sales = projections.loc['totalRevenue',('Actual',last_year)]
            ebit_inc = projections.loc['ebit',('Actual',last_year)]
            income = projections.loc['netIncomeFromContinuingOps',('Actual',last_year)]
            equity_net = projections.loc['totalStockholderEquity',('Actual',last_year)]
            valuation.EV_Sales = np.round(EV/sales,2)
            if ebit_inc > 0:
                valuation.EV_Ebit = np.round(EV/ebit_inc,2)
            else:
                valuation.EV_Ebit = 0 
            valuation.PE_Ratio = np.round(est_shares*price/income,2)
            valuation.PB_Ratio = np.round(est_shares*price/equity_net,2)
            valuation.Expected_YR_Returns = (np.exp(np.sum(stock_returns,axis=1)[0])-1)*10
            valuation.save()
        except:
            pass


    month_dict = {'Jan':1,'Feb':2,'Mar':3,'Apr':4,'May':5,'Jun':6,'Jul':7,'Aug':8,'Sep':9,'Oct':10,'Nov':11,'Dec':12}
    for stock in Stock.objects.all():
        try:
            symbol = stock.Symbol
            voltality = pred_data.loc[pred_data['Symbol']==symbol,'Voltality25'].values[0]*np.sqrt(10)
            call_data = pd.DataFrame(columns=['strikePrice','expiryDate','underlying'])
            put_data = pd.DataFrame(columns=['strikePrice','expiryDate','underlying']) 
            option_dict = option_chain(symbol)['records']['data']
            for x in option_dict:
                try:
                    items = x['CE'].items()
                    df = pd.DataFrame([[i[1] for i in items]],columns=[i[0] for i in items])
                    df = df[list(call_data.columns)]
                    call_data = call_data.append(df)
                except:
                    pass
                try:
                    items = x['PE'].items()
                    df = pd.DataFrame([[i[1] for i in items]],columns=[i[0] for i in items])
                    df = df[list(put_data.columns)]
                    put_data = put_data.append(df)
                except:
                    pass

            
            for i in range(call_data.shape[0]):
                strike_price = call_data.iloc[i]['strikePrice']
                expiryDate = call_data.iloc[i]['expiryDate']
                year = int(expiryDate[-4:])
                day = int(expiryDate[:2])
                month = month_dict[expiryDate[3:6]]
                expiry = date(year,month,day)
                t = (expiry-end).days
                stock_opt = get_history(symbol=symbol,start=start_opt,end=end,option_type="CE",strike_price=strike_price,
                        expiry_date=expiry)
                current_price = stock_opt.iloc[-1]['Settle Price']
                stock_price = stock.EOD_Price
                d1 = (np.log(stock_price/strike_price)+(3.5/100+voltality**2/2)*t/360)/(voltality*np.sqrt(t/360))
                d2 = d1 - voltality*np.sqrt(t/360)
                delta = norm.cdf(d1)*(stock.Expected_Price-stock.EOD_Price)
                theta = (stock_price*norm.pdf(d1)*voltality/(2*np.sqrt(t/360))+3.5/100*strike_price*np.exp(-3.5/100*t/360)*norm.cdf(d2))/360
                expected_price = max(current_price + delta - theta,max(stock.Expected_Price-strike_price,0))
                if Option.objects.filter(Stock=stock,Expiry=expiry,Opt_Type="CE",Strike=strike_price).exists():
                    opt = Option.objects.get(Stock=stock,Expiry=expiry,Opt_Type="CE",Strike=strike_price)
                else:
                    opt = Option()
                    opt.Stock = stock
                    opt.Expiry = expiry 
                    opt.Strike = strike_price
                    opt.Opt_Type = "CE"
                opt.EOD_Price = round(current_price,2)
                opt.Expected_Price = round(expected_price,2)
                opt.save()

            for i in range(put_data.shape[0]):
                strike_price = put_data.iloc[i]['strikePrice']
                expiryDate = put_data.iloc[i]['expiryDate']
                year = int(expiryDate[-4:])
                day = int(expiryDate[:2])
                month = month_dict[expiryDate[3:6]]
                expiry = date(year,month,day)
                t = (expiry-end).days
                stock_opt = get_history(symbol=symbol,start=start_opt,end=end,option_type="PE",strike_price=strike_price,
                        expiry_date=expiry)
                current_price = stock_opt.iloc[-1]['Settle Price']
                stock_price = stock.EOD_Price
                d1 = (np.log(stock_price/strike_price)+(3.5/100+voltality**2/2)*t/360)/(voltality*np.sqrt(t/360))
                d2 = d1 - voltality*np.sqrt(t/360)
                delta = (norm.cdf(d1)-1)*(stock.Expected_Price-stock.EOD_Price)
                theta = (stock_price*norm.pdf(d1)*voltality/(2*np.sqrt(t/360))-3.5/100*strike_price*np.exp(-3.5/100*t/360)*(1-norm.cdf(d2)))/360
                expected_price = max(current_price + delta - theta,max(-stock.Expected_Price+strike_price,0))
                if Option.objects.filter(Stock=stock,Expiry=expiry,Opt_Type="PE",Strike=strike_price).exists():
                    opt = Option.objects.get(Stock=stock,Expiry=expiry,Opt_Type="PE",Strike=strike_price)
                else:
                    opt = Option()
                    opt.Stock = stock
                    opt.Expiry = expiry 
                    opt.Strike = strike_price
                    opt.Opt_Type = "PE"
                opt.EOD_Price = round(current_price,2)
                opt.Expected_Price = round(expected_price,2)
                opt.save()

        except:
            pass 

    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()
