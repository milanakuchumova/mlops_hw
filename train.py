import dvc.api
import pandas as pd

with dvc.api.open('data/train_data.csv',
                  repo='https://github.com/milanakuchumova/mlops_hw.git') as f:
    
    df = pd.read_csv(f)
    print(df.head())