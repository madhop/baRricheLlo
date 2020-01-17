import pandas as pd
import numpy as np

def f(a):
    if np.absolute(a) > 1:
        return -a**5
    else:
        return 0

def offroad_penalty(dataset):
    trackPos = np.array(dataset['trackPos'].values)
    trackPos = np.reshape(trackPos, (len(trackPos),-1))
    penalty = np.apply_along_axis(f, 1, trackPos)
    return penalty


if __name__ == '__main__':
    dataset = pd.DataFrame([5, 1,-1,3,-10], columns=['trackPos'])
    print(offroad_penalty(dataset))
