#!/usr/bin/env python
# coding: utf-8

# In[104]:


import pandas as pd
from scipy.stats import chi2_contingency
from scipy.stats import f_oneway
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,precision_score,recall_score,f1_score,roc_curve,roc_auc_score
import numpy as np
import matplotlib.pyplot as plt


# In[62]:


ms_df=pd.read_csv('Microsoft_Data.csv')## Reading Dataset

cat_cols=[col for col in ms_df.columns if ms_df[col].dtype=='object'] 
cont_cols=[col for col in ms_df.columns if ms_df[col].dtype=='int64' or ms_df[col].dtype=='float64']


# In[63]:


corr_cols_df=ms_df.copy()##creating copy of DF for correlated columns and using this DFrame
corr_cat_cols=[col for col in corr_cols_df.columns if corr_cols_df[col].dtype=='object'] 
corr_cont_cols=[col for col in corr_cols_df.columns if corr_cols_df[col].dtype=='int64' or corr_cols_df[col].dtype=='float64']


# In[64]:


for col in corr_cont_cols:###Relacing MIss values with mean for total column just to avoid nan in finding p_values
    corr_cols_df[col].fillna(corr_cols_df[col].mean(),inplace=True)
for col in corr_cat_cols:
    corr_cols_df[col].fillna(corr_cols_df[col].mode(),inplace=True)


# In[65]:


def ch_sqr(col):#### Chi square Logic
    contigency_table=pd.crosstab(corr_cols_df[col],corr_cols_df['HasDetections'])
    res=chi2_contingency(contigency_table)
    return res[1],col

data=[ch_sqr(col) for col in corr_cat_cols]

cat_chi_sq_df=pd.DataFrame(data,columns =['p_value', 'col_name'])
corr_catg_cols=cat_chi_sq_df[cat_chi_sq_df['p_value'] < 0.05]
corr_catg_col_names=list(corr_catg_cols['col_name']) # Correlated catg columns


# In[66]:


catg=corr_cols_df['HasDetections'].value_counts().index.tolist()

def anova(col):### Anova logic
    res=f_oneway(*[corr_cols_df[corr_cols_df['HasDetections']==cat][col]for cat in catg])
    return res[1],col

cont_data=[anova(col) for col in corr_cont_cols]
cont_data

cont_anova_df=pd.DataFrame(cont_data,columns =['value', 'col_name'])
#cont_anova_df['value']=cont_anova_df['value'].fillna(0)
#cont_anova_df


# In[67]:


corr_cont_cols=cont_anova_df[cont_anova_df['value'] < 0.05]
corr_cont_col_names=list(corr_cont_cols['col_name'])
len(corr_cont_col_names)## correlated cont columns


# In[68]:


len(corr_catg_col_names)## correlated catg cols


# In[69]:


tot_corr_col_names=corr_catg_col_names + corr_cont_col_names
#tot_corr_col_names## Total correlated columns


# In[70]:


#list(np.intersect1d(ms_df_cols, tot_corr_col_names))
mainfo_df=ms_df.copy()## Creating copy of DF for deleting unwanted columns
mainfo_df.drop(columns=tot_corr_col_names,inplace=True)## Now this DF contains unwanted columns so that it can pass to next step
ls_cols_remove=list(mainfo_df.columns)


# In[71]:


ms_df.drop(columns=ls_cols_remove,inplace=True) ## Unwanted cols deleting and having only correlated cols
ms_df## only Correlated cols in main DF


# In[73]:


y=ms_df['HasDetections']

#ms_df.drop(columns=['HasDetections'],inplace=True)m
y


# In[74]:


df=pd.DataFrame({'col_name':ms_df.columns,'na_cnt':ms_df.isnull().sum(),'pc_cnt':(ms_df.isnull().sum()/ms_df.shape[0])*100})


col_gt50_ls=list(df[df['pc_cnt']>50]['col_name'])

ms_df.drop(columns=col_gt50_ls,inplace=True)

x_train,x_test,y_train,y_test=train_test_split(ms_df,y,random_state=99,test_size=0.2)

main_con_cols=[col for col in ms_df.columns if ms_df[col].dtype=='int64' or ms_df[col].dtype=='float64']
main_cat_cols=[col for col in ms_df.columns if ms_df[col].dtype=='object']


# In[75]:


for col in main_con_cols:
    x_train[col].fillna(x_train[col].mean(),inplace=True)
    x_test[col].fillna(x_train[col].mean(),inplace=True)
    
for col in main_cat_cols:
    x_train[col].fillna(x_train[col].mode()[0],inplace=True)
    x_test[col].fillna(x_train[col].mode()[0],inplace=True)


# In[80]:


scaler=StandardScaler()

for col in main_con_cols:
    x_train[col]=scaler.fit_transform(np.array(x_train[col]).reshape(-1,1))
    x_test[col]=scaler.transform(np.array(x_test[col]).reshape(-1,1))


# In[84]:


cat_encd_train=pd.get_dummies(x_train[main_cat_cols])
cat_encd_test=pd.get_dummies(x_test[main_cat_cols])


# In[85]:


cat_encd_train_final,cat_encd_test_final=cat_encd_train.align(cat_encd_test,join='inner',axis=1)
cat_encd_test_final###aligning train & test data one hot encoded catg columns due to unqual no of columns i.e no of cilumns would differ for that we align to get same


# In[86]:


x_train_final=pd.concat([x_train[main_con_cols],cat_encd_train_final],axis=1)


# In[87]:


x_test_final=pd.concat([x_test[main_con_cols],cat_encd_test_final],axis=1)


# In[92]:


logreg=LogisticRegression()
logreg.fit(x_train_final,y_train)


# In[94]:


y_test_pred=logreg.predict(x_test_final)


# In[96]:


confusion_matrix(y_test,y_test_pred)


# In[97]:


precision_score(y_test,y_test_pred)


# In[98]:


recall_score(y_test,y_test_pred)


# In[99]:


f1_score(y_test,y_test_pred)


# In[102]:


pos_probs=logreg.predict_proba(x_test_final)[::,1]
fpr,tpr,thresold=roc_curve(y_test,pos_probs)
plt.plot(fpr,tpr)
plt.xlabel('False Postive Rate')
plt.ylabel('True Postive Rate')
plt.title('AUC Curve')


# In[105]:


roc_auc_score(y_test,pos_probs)

