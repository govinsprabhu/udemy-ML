#Upper confidence bound

# importing the libraries
import numpy as np
import matplotlib as plt
import pandas as pd

# importing the data set
# CTR - Click Through Rate
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# Implement the UCB 

import math
d = 10
N = 10000
number_of_selection = [0] * d
sum_of_reward = [0] * d
ads_selected = []
total_reward = 0
for n in range(0,10000):
    max_upper_bound = 0
    ad = 0
    for i in range(0,d):
        if number_of_selection[i] > 0:
            average_reward = sum_of_reward[i]/number_of_selection[i]
            delta_i = math.sqrt(3/2 * math.log(n + 1)/ number_of_selection[i])
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e400
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i
    ads_selected.append(ad)
    number_of_selection[ad] = number_of_selection[ad] + 1
    reward = dataset.values[n, ad]
    sum_of_reward[ad] = sum_of_reward[ad] + reward
    total_reward = total_reward + reward

plt.hist(ads_selected)
plt.title('Histogram of ads selection')
plt.xlabel('Ads')
plt.ylabel('Number of times each ads was selected')
plt.show()
