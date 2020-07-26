import pandas as pd

def macd(data, shortPeriod=12, longPeriod=26):
    ema_12 = data.ewm(span=shortPeriod, adjust=False, ignore_na=False).mean()
    ema_26 = data.ewm(span=longPeriod, adjust=False, ignore_na=False).mean()
    return ema_12 - ema_26


def rsi(data, look_back_period=14):
    daily_ret = data / data.shift(1) - 1
    up = daily_ret.copy()
    down = up.copy()

    for i in up.columns:
        up[i].loc[up[i] < 0] = 0
        down[i].loc[down[i] > 0] = 0
    down = abs(down)

    dic_up = {}
    dic_down = {}
    for i in up.columns:
        dic_up[i] = [up[:look_back_period + 1][i].mean()]
        dic_down[i] = [down[:look_back_period + 1][i].mean()]

    for i in range(look_back_period + 1, len(daily_ret)):
        for j in up.columns:
            dic_up[j].append((dic_up[j][-1] * (look_back_period - 1) + up.iloc[i][j]) / look_back_period)
            dic_down[j].append((dic_down[j][-1] * (look_back_period - 1) + down.iloc[i][j]) / look_back_period)

    pd.DataFrame.from_dict(dic_up).set_index(daily_ret.index[look_back_period:])
    rsi = 100 - (100 / (1 + pd.DataFrame.from_dict(dic_up).set_index(daily_ret.index[look_back_period:]) / pd.DataFrame.from_dict(
        dic_down).set_index(daily_ret.index[look_back_period:])))
    return rsi


