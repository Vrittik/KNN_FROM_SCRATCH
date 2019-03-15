import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from matplotlib import style
from collections import Counter
import warnings

style.use('fivethirtyeight')

dataset={'k':[[1,2],[2,3],[3,1]] , 'r':[[6,5],[7,7],[8,6]]}
new_features=[1,3]


[ [ plt.scatter(ii[0],ii[1] , s=100 , color=i ) for ii in dataset[i] ] for i in dataset ]   #Here we took x and y coordinates
#  as the values sprecified in the groups as (1,2) ,(2,3) etc. This is done by running a loop over data set first 
# i.e represented by lettter i and further classification is done as ii which represents the points.
plt.scatter(new_features[0],new_features[1] , s=150 , color='green')
plt.show()


def k_nearest_neighbors (data , predict , k=3):
    if len(data)>=k:
        warnings.warn('K is set to a value less than total voting groups!')
    distances=[]
    for group in data:
        for features in data[group]:
            euclidean_distance=np.linalg.norm(np.array(features)-np.array(predict))
            distances.append([euclidean_distance,group])
    votes = [i[1] for i in sorted(distances)[:k]] # iterate the minimum distances for k distances only
    vote_result=Counter(votes).most_common(1)[0][0] #calculate out the mlost common group in the k minimum distances
    return vote_result

result = k_nearest_neighbors (dataset , new_features , k=3)

print("Your observation lies in" ,result, "group")

[ [ plt.scatter(ii[0],ii[1] , s=100 , color=i ) for ii in dataset[i] ] for i in dataset ]   #Here we took x and y coordinates
#  as the values sprecified in the groups as (1,2) ,(2,3) etc. This is done by running a loop over data set first 
# i.e represented by lettter i and further classification is done as ii which represents the points.
plt.scatter(new_features[0],new_features[1] , s=150 , color=result)
plt.show()
      
            
            