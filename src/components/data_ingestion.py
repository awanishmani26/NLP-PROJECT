import os
import pandas as pd
from sklearn.model_selection import train_test_split


class DataIngestion:

    def __init__(self):

        self.train_path = "artifacts/train.csv"
        self.test_path = "artifacts/test.csv"

    def initiate_data_ingestion(self):

        df = pd.read_csv("notebook\data\\all_kindle_review .csv")

        df = df[['reviewText','rating']]

        df.dropna(inplace=True)

        df['sentiment'] = df['rating'].apply(lambda x: 1 if x >=3 else 0)

        df = df[['reviewText','sentiment']]

        train_set,test_set = train_test_split(df,test_size=0.2,random_state=42)

        os.makedirs("artifacts",exist_ok=True)

        train_set.to_csv(self.train_path,index=False)

        test_set.to_csv(self.test_path,index=False)

        return self.train_path,self.test_path