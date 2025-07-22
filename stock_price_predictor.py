import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
import pandas as pd

class SP500Predictor:
    def __init__(self, start_date="1990-01-01"):
        self.ticker = "^GSPC"
        self.start_date = start_date
        self.model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)
        self.data = self.load_data()
        self.predictors = []
    
    def load_data(self):
        sp500 = yf.Ticker(self.ticker).history(period="max")
        sp500.drop(columns=["Dividends", "Stock Splits"], inplace=True)
        sp500["Tomorrow"] = sp500["Close"].shift(-1)
        sp500["Target"] = (sp500["Tomorrow"] > sp500["Close"]).astype(int)
        sp500 = sp500.loc[self.start_date:].copy()
        return sp500

    def add_features(self, horizons=[2, 5, 60, 250, 1000]):
        for horizon in horizons:
            rolling_avg = self.data["Close"].rolling(horizon).mean()
            ratio_col = f"Close_Ratio_{horizon}"
            trend_col = f"Trend_{horizon}"

            self.data[ratio_col] = self.data["Close"] / rolling_avg
            self.data[trend_col] = self.data["Target"].shift(1).rolling(horizon).sum()

            self.predictors += [ratio_col, trend_col]
        
        self.data.dropna(inplace=True)

    def train_and_predict(self, train, test):
        self.model.fit(train[self.predictors], train["Target"])
        probs = self.model.predict_proba(test[self.predictors])[:, 1]
        preds = (probs >= 0.6).astype(int)
        return pd.Series(preds, index=test.index, name="Predictions")

    def backtest(self, start=2500, step=250):
        all_predictions = []

        for i in range(start, self.data.shape[0], step):
            train = self.data.iloc[0:i].copy()
            test = self.data.iloc[i:(i+step)].copy()
            preds = self.train_and_predict(train, test)
            combined = pd.concat([test["Target"], preds], axis=1)
            all_predictions.append(combined)

        return pd.concat(all_predictions)

    def evaluate(self, predictions):
        accuracy = precision_score(predictions["Target"], predictions["Predictions"])
        print(f"Precision Score: {accuracy:.4f}")

        predictions["Correct"] = predictions["Target"] == predictions["Predictions"]
        predictions["Correct"].rolling(100).mean().plot(figsize=(15, 6))
        plt.title("Rolling Prediction Accuracy (window=100 days)")
        plt.ylabel("Accuracy")
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    predictor = SP500Predictor()
    predictor.add_features()
    predictions = predictor.backtest()
    predictor.evaluate(predictions)
