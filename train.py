import pandas as pd
import fire
import dvc.api

def read_data(filename: str):
    df = pd.read_csv(filename)
    

def main():
    url = dvc.api.get_url('train_data.csv',
        repo='gdrive://1glMslxpGLmz_GwYXUdiy5hr5C2vEIW_Q/files')
    print(url)


if __name__ == '__main__':
    main()
    # fire.Fire(main)