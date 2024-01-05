import dvc.api

df = dvc.api.read('train_data.csv',
                  repo='gdrive://1glMslxpGLmz_GwYXUdiy5hr5C2vEIW_Q')
print(df)