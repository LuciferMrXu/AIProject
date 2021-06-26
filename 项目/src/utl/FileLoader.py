import pandas as pd
import numpy as np

def load2DfHead(path, head=5, sep=","):
    n = 0
    ls = []
    with open(path) as f:
        for line in f:
            if n == 0:
                columns = line.split(sep)
                n += 1
            else:
                ls.append(line.split(sep))
                n += 1
            if n >= head:
                break
    arr = np.array(ls)
    df = pd.DataFrame(arr,columns=columns)
    return df

