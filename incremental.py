#main function
def main():
  start = time.time()
  df = pd.read_csv('/content/Test.csv')
  print(df.head())

  #applying MinMaxScaler
  #scaler = MinMaxScaler()
  #df = scaler.fit_transform(df)
  #df_scaled = pd.DataFrame(df,columns = ['Faculty','Department','Conference_Details','University'])
  #print(df_scaled)
  df_incremental = incremental_cluster(df)
  print(df_incremental)
  visualize_increment(df_incremental)
  print('Execution time {} seconds'.format(time.time()-start))

if __name__=='__main__':
  main()