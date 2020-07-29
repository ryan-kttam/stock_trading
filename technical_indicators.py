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


def smoothing(data, look_back_period):
    # applying this equation: y_t = (1-a)*y_(t-1) + x_t/n, where a = 1/n
    # this is equivalent to wilder smoothing
    result = data.copy()
    result = result.rolling(look_back_period).mean()
    result[look_back_period:] = data[look_back_period:]
    return result.ewm(alpha=1 / look_back_period, adjust=False).mean()


def adx(close, high, low, look_back_period, modified_version=False):
    # This function will return two dataframes: ADX and (plus_di-minus_di)

    # calculate the true range
    tr = pd.DataFrame(index=close.index)
    tmp = pd.concat([high - low,
                     abs(high - close.shift(1)),
                     abs(low - close.shift(1))], axis=1)
    for i in close.columns:
        tr[i] = tmp[i].max(axis=1)

    direction_movement_high = high - high.shift(1)
    direction_movement_low = low.shift(1) - low
    direction_movement_high[direction_movement_high < 0] = 0
    direction_movement_low[direction_movement_low < 0] = 0
    direction_movement_high[~(direction_movement_high > direction_movement_low)] = 0
    direction_movement_low[~(direction_movement_low > direction_movement_high)] = 0

    smoothed_tr = smoothing(tr, look_back_period)
    smoothed_dmh = smoothing(direction_movement_high, look_back_period)
    smoothed_dml = smoothing(direction_movement_low, look_back_period)

    plus_di = smoothed_dmh / smoothed_tr * 100
    minus_di = smoothed_dml / smoothed_tr * 100

    # Calculating DX and ADX
    dx = abs(plus_di - minus_di) / abs(plus_di + minus_di) * 100
    adx = smoothing(dx, look_back_period)
    signals = (plus_di - minus_di)

    if modified_version:
        signals_2 = signals.copy()
        for i in signals.columns:
            signals[i][signals[i] > 0] = 1
            signals[i][signals[i] < 0] = -1
        return (adx * signals), signals_2
    else:
        return adx, signals
