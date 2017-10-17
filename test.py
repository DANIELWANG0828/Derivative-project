# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 13:40:52 2017

@author: dell
"""
import numpy as np
import pandas as pd
import alphalens
import datetime
factor=pd.read_csv('factors.csv')
price=pd.read_csv('price_init.csv')
fac=factor[['CashRateOfSales','BLEV','CurrentRatio',]]sector_names={0: u'\u80fd\u6e90',
 1: u'\u539f\u6750\u6599',
 2: u'\u5de5\u4e1a',
 3: u'\u53ef\u9009\u6d88\u8d39',
 4: u'\u4e3b\u8981\u6d88\u8d39',
 5: u'\u533b\u836f\u536b\u751f',
 6: u'\u91d1\u878d\u5730\u4ea7',
 7: u'\u4fe1\u606f\u6280\u672f',
 8: u'\u7535\u4fe1\u4e1a\u52a1',
 9: u'\u516c\u7528\u4e8b\u4e1a'}

ticker_sector=pd.read_csv('ticker_sector.csv')
ticker_sector=ticker_sector.set_index('secID').to_dict()
ticker_sector=ticker_sector['code']
factor=factor[factor.columns[1:]]
factor['tradeDate']=factor['tradeDate'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d'))
factor=factor.set_index(['tradeDate','secID'])

price['tradeDate']=price['tradeDate'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d'))
price=price.set_index('tradeDate')

def test(factor,price,ticker_sector,sector_names,name):
    
  
    

    factor=factor[name]


    clean= alphalens.utils.get_clean_factor_and_forward_returns(factor,price,groupby=ticker_sector, groupby_labels=sector_names, periods=(10,20,60,120),quantiles=5)
    return clean
    

def test_raw(factor,price,ticker_sector,sector_names):
 
    factor=factor[factor.columns[1:]]
    factor['tradeDate']=factor['tradeDate'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d'))
    factor=factor.set_index(['tradeDate','secID'])
    factor=factor.iloc[:,0]


    price=pd.pivot(index='tradeDate',columns='secID',values='price')
    price['tradeDate']=price['tradeDate'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d'))
    price=price.set_index('tradeDate')
    

    clean= alphalens.utils.get_clean_factor_and_forward_returns(factor,price,groupby=ticker_sector, groupby_labels=sector_names, periods=(1,5,10,20,60,120,240),quantiles=10)
    return alphalens.tears.create_full_tear_sheet(clean) 
    alphalens.tears.create_full_tear_sheet(clean)
    
    
    clean= alphalens.utils.get_clean_factor_and_forward_returns(DebtsAssetRatio,price,groupby=ticker_sector, groupby_labels=sector_names, periods=(10,20,60,120),quantiles=5)
    alphalens.tears.create_full_tear_sheet(clean)

def standardize(obj):
    obj=(obj-obj.mean())/np.std(obj)
    return obj
    ic_weight_df
    
    

def ic_get(fac):
    fac_data=factor[fac]
    clean= alphalens.utils.get_clean_factor_and_forward_returns(fac_data,price,groupby=ticker_sector, groupby_labels=sector_names, periods=(5,10,20),quantiles=5)
    ic = alphalens.performance.factor_information_coefficient(clean, group_adjust=False)
    ic=pd.DataFrame(ic[5],columns=[5])
    ic.columns=[fac]
    return ic
    
def weight_get():
    ic_BLEV=ic_get(BLEV)
    ic_BLEV=ic_get(BLEV)
    ic_DebtsAssetRatio=ic_get(DebtsAssetRatio)
    ic
    ic
    
    BLEV.index.levels[1].tolist()
    n = 120 
    ic_weight_df = pd.DataFrame(index=ic_df.index, columns=ic_df.columns)
    for dt in ic_df.index:
        ic_dt = ic_df[ic_df.index<dt].tail(n)
        if len(ic_dt) < n:
            continue
        
    ic_cov_mat = np.mat(np.cov(ic_dt.T.as_matrix()).astype(float)) 
    inv_ic_cov_mat = np.linalg.inv(ic_cov_mat)
    weight = inv_ic_cov_mat*np.mat(ic_dt.mean()).reshape(len(inv_ic_cov_mat),1)
    weight = np.array(weight.reshape(len(weight),))[0]
    ic_weight_df.ix[dt] = weight/np.sum(weight)
    
    
    for dt in ic_weight_df.index:
        for stk in universe:
            fac.loc[[dt,stk]]['fac_ulti']=fac.loc[[dt,stk]]['BLEV']*ic_weight_df['BLEV'][dt]
    


for dt in fac.index.levels[0]:
    for stk in fac.index.levels[1]:
        fac.T[dt,stk]['fac_syn']=fac.T[dt,stk]['DebtsAssetRatio']*ic['DebtsAssetRatio'][dt]+fac.T[dt,stk]['CurrentRatio']*ic['CurrentRatio'][dt]+fac.T[dt,stk]['BLEV']*ic['blev'][dt]+fac.T[dt,stk]['CashRateOfSales']*ic['crs'][dt]
    

fac=factor[['CashRateOfSales','BLEV','CurrentRatio','DebtsAssetRatio']]

tem=pd.DataFrame(index=fac.index,columns=['fac_syn'])

fac=pd.concat([fac,tem],axis=1)










fac_data=factor['CashRateOfSales']
clean= alphalens.utils.get_clean_factor_and_forward_returns(fac_data,price,groupby=ticker_sector, groupby_labels=sector_names, periods=(5,10,20),quantiles=5)
ic_crs = alphalens.performance.factor_information_coefficient(clean, group_adjust=False)
ic_crs=pd.DataFrame(ic[5],columns=[5])
ic_crs.columns=['CashRateOfSales']

fac_data=factor['CashRateOfSales']
clean= alphalens.utils.get_clean_factor_and_forward_returns(fac_data,price,groupby=ticker_sector, groupby_labels=sector_names, periods=(5,10,20),quantiles=5)
ic_crs = alphalens.performance.factor_information_coefficient(clean, group_adjust=False)
ic_crs=pd.DataFrame(ic[5],columns=['CashRateOfSales'])

fac_data=factor['CashRateOfSales']
clean= alphalens.utils.get_clean_factor_and_forward_returns(fac_data,price,groupby=ticker_sector, groupby_labels=sector_names, periods=(5,10,20),quantiles=5)
ic_crs = alphalens.performance.factor_information_coefficient(clean, group_adjust=False)

ic_crs.head()
ic_crs[5]
ic_crs=pd.DataFrame(ic_crs[5],columns=[5])
ic_crs.columns=['crs']
del ic
ic_crs.head()
fac_data=factor['BLEV']
clean= alphalens.utils.get_clean_factor_and_forward_returns(fac_data,price,groupby=ticker_sector, groupby_labels=sector_names, periods=(5,10,20),quantiles=5)
ic_blev = alphalens.performance.factor_information_coefficient(clean, group_adjust=False)
ic_blev=pd.DataFrame(ic_blev[5],columns=[5])
ic_blev.columns=['BLEV']

ic_blev.columns=['blev']
fac_data=factor['CurrentRatio']
clean= alphalens.utils.get_clean_factor_and_forward_returns(fac_data,price,groupby=ticker_sector, groupby_labels=sector_names, periods=(5,10,20),quantiles=5)
ic_cr = alphalens.performance.factor_information_coefficient(clean, group_adjust=False)
ic_cr=pd.DataFrame(ic_cr[5],columns=[5])
ic_cr.columns=['CurrentRatio']

fac_data=factor['DebtsAssetRatio']
clean= alphalens.utils.get_clean_factor_and_forward_returns(fac_data,price,groupby=ticker_sector, groupby_labels=sector_names, periods=(5,10,20),quantiles=5)
ic_dar = alphalens.performance.factor_information_coefficient(clean, group_adjust=False)
ic_dar=pd.DataFrame(ic_dar[5],columns=[5])
ic_dar.columns=['DebtsAssetRatio']

ic=pd.merge(ic_crs.reset_index(),ic_blev.reset_index())
ic.head()
ic=pd.merge(ic.reset_index(),ic_dar.reset_index())
ic.head()
del ic['index']
ic.head()
ic=pd.merge(ic,ic_cr.reset_index())
ic.head()
ic.set_index('date')
ic=ic.set_index('date')


for dt in fac.index.levels[0]:






    tem=np.dot(ic_weight_df.loc[dt].values[:,:],fac.T[dt].values[:,:])



    fac_final={}
for dt in fac.index.levels[0]:

    tem=np.dot(ic_weight_df.loc[dt].values[:],fac.T[dt].values[:,:])
    tem=pd.DataFrame(tem)
    tem.index=['fac_syn']
    tem.columns=fac.T[dt].index
    fac_final[dt]=tem
    fac_final=pd.DataFrame(fac_final)