__author__ = 'jaap.meijer'

import numpy as np
import pandas as pd

def interpolate_1h(df, n):

    # df = pd.DataFrame.from_csv('./CRAF/waves/wind.txt')
    # if strings have to be maintained
    # df = df[50:65].resample('H', how='first', limit=5)

    df = df.resample('H')
    #df = df.interpolate(limit=2)

    headers = list(df.columns.values)

    for i in range(0,len(df)-1,n):

        if ~np.isnan(df[headers[0]][i + n]):
            a1 = (df[headers[0]][i + n] - df[headers[0]][i]) / n
            a2 = (df[headers[1]][i + n] - df[headers[1]][i]) / n
            df[headers[0]][i + 1] = a1 * (df.index.hour[i + 1] - df.index.hour[i]) + df[headers[0]][i]
            df[headers[0]][i + 2] = a1 * (df.index.hour[i + 2] - df.index.hour[i]) + df[headers[0]][i]
            df[headers[1]][i + 1] = a2 * (df.index.hour[i + 1] - df.index.hour[i]) + df[headers[1]][i]
            df[headers[1]][i + 2] = a2 * (df.index.hour[i + 2] - df.index.hour[i]) + df[headers[1]][i]

    return df