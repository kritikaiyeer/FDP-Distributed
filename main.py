import pandas as pd
from Datapreprocess import *
from cfba import *
import os
import boto3
from botocore.exceptions import ClientError

import time

access_key ='YOUR_KEY'
access_secret = 'YOUR_KEY'
bucket_name = 'YOUR_BUCKET'

def main():
  start = time.time()
  df = pd.read_csv('record.csv')
  df = cleanData(df)
  df_copy = df.copy()
  df = encodeData(df)
  print(df.head())
  train,test,df1,df2 = scale(df)
  df_basic,df_train = basic_cluster_lone(train,df1)
  df_incremental , dftest = incremental_cluster(test,df2)
  merge_basic_incremental1 = mergefile_graph(df_basic,df_incremental)
  mergefile_representative(df_train,dftest)
  visualize(df_basic,df_incremental,merge_basic_incremental1)
  client_s3 = boto3.client(
    's3',
    aws_access_key_id=access_key,
    aws_secret_access_key=access_secret
  )
  print('S3 Bucket Loaded....... ')
  print('Uploading File....... ')
  client_s3.upload_file('record.csv', bucket_name, 'record.csv')
  
  print('Execution time {} seconds'.format(time.time()-start))

if __name__=='__main__':
  main()