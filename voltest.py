from AlgoAPI import AlgoAPIUtil, AlgoAPI_Backtest
from math import floor
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
from sklearn.preprocessing import RobustScaler
import lightgbm as lgb
from scipy.optimize import minimize
from sklearn.linear_model import BayesianRidge

class AlgoEvent:
    def __init__(self):
        self.positions = {}
        self.lasttradetime = datetime(2000,1,1)
        self.current_day = None
        self.prices = {}
        self.instruments = []
        self.volatilities = {}
        self.rv = {}
        self.model = HybridHARModel(boot_weekly=0.02, boot_monthly=0.02, window_size=60)
        self.min_data_points = 2

    def start(self, mEvt):
        self.instruments = mEvt['subscribeList']
        self.evt = AlgoAPI_Backtest.AlgoEvtHandler(self, mEvt)
        for instrument in self.instruments:
            self.prices[instrument] = []
            self.volatilities[instrument] = []
            self.rv[instrument] = []
            self.positions[instrument] = 0
        self.evt.start()

    def on_marketdatafeed(self, md, ab):
        current_time = md['timestamp']
        dt_obj = datetime.strptime(current_time, "%Y-%m-%d %H:%M:%S.%f")
        current_date = dt_obj.strftime("%Y/%m/%d")
        current_time = datetime.strptime(current_time, "%Y-%m-%d %H:%M:%S.%f")
        
        if (current_time - self.lasttradetime) < timedelta(minutes=5):
            return
        self.lasttradetime = current_time


        for instrument in self.instruments:
            if instrument not in md:
                return
            self.prices[instrument].append(md[instrument]['lastPrice'])
            if len(self.prices[instrument]) >= 2:
                self.realised_volatility(self.prices[instrument], instrument)

        if (current_date != self.current_day):
            for instrument in self.instruments:
                forecast = self.get_volatility_forecast(instrument)
                if forecast is not None:
                    self.volatilities[instrument].append(forecast)
                    current_position = self.positions[instrument]
                    order_volume = self.calculate_position_size(instrument, forecast, ab, current_position)
                    if order_volume > 0:
                        self.submitOrder(instrument, order_volume, buysell='buy')
                    elif order_volume < 0:
                        self.submitOrder(instrument, abs(order_volume), buysell='sell')

                self.rv[instrument].append(0)
                self.prices[instrument] = []
            self.current_day = current_date


                
    def realised_volatility(self, prices, instrument):
        rv = 0
        for i in range(len(prices)-1):
            rt = (np.log(prices[i+1] / prices[i]))**2
            rv += rt
        
        self.rv[instrument][-1] = rv

        return rv
    

    def get_volatility_forecast(self, instrument):
        if len(self.rv[instrument]) < self.min_data_points:
            return None

        forecast = self.model.update(self.rv[instrument])
        
        return forecast

    def calculate_position_size(self, instrument, volatility, ab, current_position):
        n = len(self.instruments)
        c = 0.015
        day_leverage = c / volatility
        balance = ab['availableBalance']
        instrument_exposure = (1/n) * day_leverage * balance
        unit_price = self.prices[instrument][-1]
        target_volume = instrument_exposure / unit_price
        order_volume = target_volume - current_position
        if abs(order_volume) < 1:
            return 0
        return floor(order_volume)

    def submitOrder(self, instrument, volume, buysell):
        if volume == 0:
            return
            
        order = AlgoAPIUtil.OrderObject(
            instrument=instrument,
            volume=volume,
            openclose='open',
            buysell=buysell,
            ordertype=0,       # 0=market_order
        )
        self.evt.sendOrder(order)

    def on_bulkdatafeed(self, md, ab):
        pass

    def on_orderfeed(self, of):
        pass

    def on_dailyPLfeed(self, pl):
        pass

    def on_openPositionfeed(self, op, oo, uo):
        for instrument in self.instruments:
            if instrument in op:
                self.positions[instrument] = op[instrument]['netVolume']
            else:
                self.positions[instrument] = 0























        


class BayesianHAR:
    def __init__(self):
        self.model = BayesianRidge()
        self.is_fitted = False
    
    def fit(self, X, y):
        self.model.fit(X, y)
        self.is_fitted = True
        return self
    
    def predict(self, X):
        return self.model.predict(X)

class FeatureEngineer:
    def __init__(self):
        self.scaler = RobustScaler()
        
    def create_features(self, rv_series):
        df = pd.DataFrame({
            'rv': rv_series,
            'log_rv': np.log(rv_series)
        })
        
        df['zscore_5'] = (df['rv'] - df['rv'].rolling(5).mean()) / df['rv'].rolling(5).std()
        df['skew_22'] = df['rv'].rolling(22).skew()
        
        df['high_vol'] = (df['rv'] > df['rv'].rolling(22).quantile(0.75)).astype(int)
        
        features_to_scale = ['rv', 'log_rv', 'zscore_5']
        df[features_to_scale] = self.scaler.fit_transform(df[features_to_scale])
        
        return df.dropna()

class ResidualCorrector:
    def __init__(self):
        self.model = lgb.LGBMRegressor(
            num_leaves=7,
            learning_rate=0.01,
            n_estimators=300,
            reg_alpha=0.1
        )
    
    def fit(self, X, residuals):
        self.model.fit(X, residuals,
            eval_metric='mae')
    
    def predict(self, X):
        return self.model.predict(X)
    
class DynamicWeights:
    def __init__(self, n_models=2):
        self.weights = np.ones(n_models)/n_models
        self.cov_matrix = np.eye(n_models)
    
    def update_weights(self, recent_errors):
        def objective(w):
            return w.T @ self.cov_matrix @ w
        
        cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        bounds = [(0.1, 0.9) for _ in range(len(self.weights))]
        
        res = minimize(objective, self.weights,
                      method='SLSQP',
                      bounds=bounds,
                      constraints=cons)
        
        self.weights = res.x
        self._update_covariance(recent_errors)
    
    def _update_covariance(self, errors):
        self.cov_matrix = np.cov(errors, rowvar=False)

class HybridHARModel:
    def __init__(self, boot_weekly=0.02, boot_monthly=0.02, window_size=60):
        self.har = BayesianHAR()
        self.fe = FeatureEngineer()
        self.ml = ResidualCorrector()
        self.weights = DynamicWeights()
        self.error_history = []
        self.rv_series = []
        self.boot_weekly = boot_weekly
        self.boot_monthly = boot_monthly
        self.window_size = window_size

    def make_har_features(self, rv_series):
        rv_series = rv_series[-(self.window_size+1):] if len(rv_series) > self.window_size+1 else rv_series
        n = len(rv_series)
        X = []
        y = []
        for t in range(n-1):
            current = rv_series[t]
            n_week = min(t+1, 5)
            w_week = max(0, 1 - n_week/5)
            rolling_week = np.mean(rv_series[max(0, t-4):t+1])
            weekly = w_week * self.boot_weekly + (1-w_week) * rolling_week
            n_month = min(t+1, 22)
            w_month = max(0, 1 - n_month/22)
            rolling_month = np.mean(rv_series[max(0, t-21):t+1])
            monthly = w_month * self.boot_monthly + (1-w_month) * rolling_month
            X.append([1, current, weekly, monthly])
            y.append(rv_series[t+1])
        return np.array(X), np.array(y)

    def make_har_features_predict(self, rv_series):
        t = len(rv_series)-1
        current = rv_series[t]
        n_week = min(t+1, 5)
        w_week = max(0, 1 - n_week/5)
        rolling_week = np.mean(rv_series[max(0, t-4):t+1])
        weekly = w_week * self.boot_weekly + (1-w_week) * rolling_week
        n_month = min(t+1, 22)
        w_month = max(0, 1 - n_month/22)
        rolling_month = np.mean(rv_series[max(0, t-21):t+1])
        monthly = w_month * self.boot_monthly + (1-w_month) * rolling_month
        return np.array([[1, current, weekly, monthly]])

    def update(self, rv_series):
        self.rv_series = rv_series
        if len(rv_series) < 2:
            return None 
        X, y = self.make_har_features(np.array(rv_series))
        if len(X) < 2:
            return None
        self.har.fit(X, y)
        X_pred = self.make_har_features_predict(np.array(rv_series))
        har_pred = self.har.predict(X_pred)[0]
        residuals = y - self.har.predict(X)
        self.ml.fit(X, residuals)
        ml_pred = np.array(self.ml.predict(X_pred))[0]
        combined = self.weights.weights[0] * har_pred + self.weights.weights[1] * ml_pred
        return combined

