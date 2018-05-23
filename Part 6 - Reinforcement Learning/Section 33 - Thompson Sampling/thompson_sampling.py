#Thompson Sampling

# importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing the data set
# CTR - Click Through Rate
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# Implement the Thompson Sampling 

import random
d = 10
N = 10000
number_of_rewards_1 = [0] * d
number_of_rewards_0 = [0] * d
ads_selected = []
total_reward = 0
for n in range(0,10000):
    max_random = 0
    ad = 0
    for i in range(0,d):
        random_beta = random.betavariate(number_of_rewards_1[i] + 1, number_of_rewards_0[i] + 1)
        if random_beta > max_random:
            max_random = random_beta
            ad = i
    ads_selected.append(ad)
    reward = dataset.values[n, ad]
    if reward == 1:
        number_of_rewards_1[ad] = number_of_rewards_1[ad] + 1
    else:
        number_of_rewards_0[ad] = number_of_rewards_0[ad] + 1
    total_reward = total_reward + reward
    
    

plt.hist(ads_selected)
plt.title('Histogram of ads selection')
plt.xlabel('Ads')
plt.ylabel('Number of times each ads was selected')
plt.show()
