import pandas as pd
import math as mp
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def visualize(df_basic,df_incremental,df_merge):  

  ax = df_basic.groupby(['CNumber'])['CNumber'].count().plot.bar(title = "Basic...")
  ax.set_xlabel('Clusters')
  ax.set_ylabel('Frequency')
  plt.show()

  ax = df_incremental.groupby(['CNumber'])['CNumber'].count().plot.bar(title = "Incremental")
  ax.set_xlabel('Clusters')
  ax.set_ylabel('Frequency')
  plt.show()

  ax = df_merge.groupby(['CNumber','Cluster_Type'])['Cluster_Type'].count().unstack(0).plot.bar(stacked=True, figsize=(8, 6))
  ax.legend(loc = 'center right',bbox_to_anchor = (1.4,0.5),ncol = 1)
  plt.title('iteration 1')
  plt.xlabel('clusters')
  plt.ylabel('No of Records')
  plt.show()

# merging basic and incremental dataset
def mergefile_graph(df_basic,df_incremental):
  df_basic['Cluster_Type'] = 'Basic_cluster'
  df_incremental['Cluster_Type'] = 'Incremental_1'
  df_basic = df_basic.append(df_incremental)
  df_basic=df_basic.sort_values(by = ['CNumber'])
  df_basic.to_csv('record.csv',index = False)
  print("df_basic length", len(df_basic))
  return df_basic

#merging training and test dataset
def mergefile_representative(dftrain,dftest):
  dftrain = dftrain.append(dftest)
  dftrain = dftrain.sort_values(by = ['CNumber'])
  dftrain.to_csv('record.csv',index = False)
  #(dftrain.groupby(['CNumber'],as_index = False).mean()).to_csv('record.csv')

#basic clustering code using cfba
def basic_cluster_lone(df,df1):
  df['row_total'] = df.sum(axis = 1)
  print("after row total",df.head())
  count = 1
  closeness_val= []
  for i in range(len(df)):
    df.loc[i,'Flag']=False
    c1 = []
  

  for i in range(len(df)):
    if(df.Flag[i]==False):
      countercheck = []
      df1.loc[i,'CNumber'] = count
      df1.loc[i,'Closeness_Value'] = 0
      df.loc[i,'Flag']=True
      df.loc[i,'CNumber'] = count
      for j in range(i+1,len(df)):
        if(df.Flag[j]==False):
          
          c1 = df.row_total[i]/(df.row_total[i]+df.row_total[j])
          
          d1 = df.Faculty[i]+df.Faculty[j]
          d2=c1*d1-df.Faculty[i]
          d3 = mp.sqrt(d1*c1*(1-c1))
          prob1 = d2/d3
          c_square = mp.pow(prob1,2)
          weight = mp.sqrt(d1)
          c = c_square * weight
          
          #second feature
          col2 = df.Department[i]+df.Department[j]
          col21 = (c1*col2-df.Department[i])/mp.sqrt(col2*c1*(1-c1))
          e2 = mp.pow(col21,2)
          wei2 = mp.sqrt(col2)
          c2 = e2 * wei2
  

          #third feature
          col4 = df.University[i]+df.University[j]
          col41 = (c1*col4-df.University[i])/mp.sqrt(col4*c1*(1-c1))
          e4 = mp.pow(col41,2)
          wei4 = mp.sqrt(col4)
          c4 = e4 * wei4

          close1 = c+c2+c4
          close2 = weight+wei2+wei4
          close = close1/close2
          counter = 1
          
          if close<=1:
            df1.loc[j,'CNumber'] = count
            df1.loc[j,'Closeness_Value']=close
            df.loc[j,'Flag']=True
            df.loc[j,'CNumber']=count
            if(close < 0.0120326011250972):
                df1.loc[j,'CNumber']=counter
                df.loc[j,'CNumber']=counter
            elif(0.0120326011250972 < close < 0.221485769509811):
                 df1.loc[j,'CNumber']=counter+1
                 df.loc[j,'CNumber']=counter+1
            elif(0.221485769509811 < close < 0.706277787871182):
                 df1.loc[j,'CNumber']=counter+2
                 df.loc[j,'CNumber']=counter+2
            else:
                 df1.loc[j,'CNumber']=counter+3
            df1.to_csv('record.csv')
            
  
  df1 = df1.sort_index()
  df1 = df1.sort_values(by = 'CNumber')
  df1.to_csv('record.csv')

  #add name of csv
  df =df.drop(['Flag','row_total'],axis=1)
  
  
  return df1,df

# incremental clustering code using cfba
def incremental_cluster(dftest,df2):
  df = pd.read_csv('record.csv')
  print("test data",df.head())
  df_rep = df.iloc[:,1:]
  df_rep['row_total'] = df_rep.sum(axis =1)
  print(df_rep.head())
  whole = []
  outlier = []
  fclose=[]
  outlierclose=[]
  dftest['row_total'] = dftest.sum(axis =1)
  for i in range(len(dftest)):
    dftest.loc[i,'Flag']=False
  c1 = []
  for i in range(len(df_rep)):
    whole.append(i)
    for j in range(len(dftest)):
      if(dftest.Flag[j]==False):
        c1 = df_rep.row_total[i]/(df_rep.row_total[i]+dftest.row_total[j])
        
        d1 = df_rep.Faculty[i]+dftest.Faculty[j]
        d2=c1*d1-df_rep.Faculty[i]
        d3 = mp.sqrt(d1*c1*(1-c1))
        prob1 = d2/d3
        c_square = mp.pow(prob1,2)
        weight = mp.sqrt(d1)
        c = c_square * weight
       
        #second feature - Department
        col2 = df_rep.Department[i]+dftest.Department[j]
        col21 = (c1*col2-df_rep.Department[i])/mp.sqrt(col2*c1*(1-c1))
        e2 = mp.pow(col21,2)
        wei2 = mp.sqrt(col2)
        c2 = e2 * wei2
        

        #fourth feature
        col4 = df_rep.University[i]+dftest.University[j]
        col41 = (c1*col4-df_rep.University[i])/mp.sqrt(col4*c1*(1-c1))
        e4 = mp.pow(col41,2)
        wei4 = mp.sqrt(col4)
        c4 = e4 * wei4


        close1 = c+c2+c4
        close2 = weight+wei2+wei4
        close = close1/close2

        if close<=1:
          whole.append(j)
          df2.loc[j,'CNumber'] = df.CNumber[i]
          df2.loc[j,'Closeness Value']=close
          dftest.loc[j,'Flag']=True
          dftest.loc[j,'CNumber']=df.CNumber[i]
          #add name of csv of incremental
          df2.to_csv('record.csv')
        else:
          outlier.append(j)
          outlierclose.append(close)
    fclose.append(0)

  resultant_list = list(set(outlier)-set(whole))
  if(len(resultant_list)!=None):
    dftest.loc[resultant_list,'CNumber']=i+2
    dftest.loc[resultant_list,'Flag']=True
    df2.loc[resultant_list,'CNumber']=i+2
  df2 = df2.fillna(-1)
  df2 = df2.sort_index()
  df2 = df2.sort_values(by = 'CNumber')
  #add name of csv
  df2.to_csv('record.csv')
  dftest =dftest.drop(['Flag','row_total'],axis=1)
  return df2,dftest

def scale(pandas_df):
  features = ['Faculty','Department','Conference_Details','University']
  features_v = pandas_df[features]
  scaler = MinMaxScaler(feature_range = (0,10))
  scaler_features = scaler.fit_transform(features_v)
  print("normalised dataset with MinMaxScaler",scaler_features)
  features_train, features_test = train_test_split(scaler_features, test_size =0.2)
  train1 = pd.DataFrame(features_train,columns = ['Faculty','Department','Conference_Details','University'])
  test1 = pd.DataFrame(features_test,columns = ['Faculty','Department','Conference_Details','University'])
  print("length of training and testing data",len(train1),len(test1))
  df_inverse = scaler.inverse_transform(features_train)
  df1 = pd.DataFrame(df_inverse,columns = ['Faculty','Department','Conference_Details','University'])
  df2 = scaler.inverse_transform(features_test)
  df2 = pd.DataFrame(df2,columns = ['Faculty','Department','Conference_Details','University'])
  print("length of Inversed data",len(df1),len(df2))
  return train1,test1,df1,df2

