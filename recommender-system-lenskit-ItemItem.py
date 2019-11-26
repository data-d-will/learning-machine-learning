from lenskit import batch, topn, util
from lenskit import crossfold as xf
from lenskit.algorithms import als, item_knn
from lenskit.metrics import topn as tnmetrics

import pandas as pd
#%matplotlib inline
import matplotlib as plot

ratings = pd.read_csv('ml-100k/u.data', sep='\t',
    names=['user', 'item', 'rating', 'timestamp'])
print(ratings.head())

algo_ii = item_knn.ItemItem(20)
algo_als = als.BiasedMF(50)

#item_knn.ItemItem(20).fit(train)

""" def eval(aname, algo, train, test):
    #fittable = util.clone(algo)
    algo.fit(train)
    users = test.user.unique()
    # the recommend function can merge rating values
    recs = batch.recommend(algo, users, 100,
            topn.UnratedCandidates(train), test)
    # add the algorithm
    recs['Algorithm'] = aname
    return recs """

all_recs = []
test_data = []
for train, test in xf.partition_users(ratings[['user', 'item', 'rating']], 5, xf.SampleFrac(0.2)):
    test_data.append(test)
    
    # ItemItem
    # eval('ItemItem', algo_ii, train, test)
    #fittable = util.clone(algo)
    algo_ii.fit(train)
    users = test.user.unique()
    # the recommend function can merge rating values
    recs = batch.recommend(algo_ii, users, 100, topn.UnratedCandidates(train), test)
    # add the algorithm
    recs['Algorithm'] = 'ItemItem'

    all_recs.append(recs)

    # ItemItem
    # all_recs.append(eval('ALS', algo_als, train, test))
    #fittable = util.clone(algo)
    algo_als.fit(train)
    users2 = test.user.unique()
    # the recommend function can merge rating values
    recs2 = batch.recommend(algo_als, users2, 100, topn.UnratedCandidates(train), test)
    # add the algorithm
    recs2['Algorithm'] = 'ALS'

    all_recs.append(recs2)
    

all_recs = pd.concat(all_recs, ignore_index=True)
all_recs.head()

#10
user_dcg = all_recs.groupby(['Algorithm', 'user']).rating.apply(tnmetrics.dcg)
user_dcg = user_dcg.reset_index(name='DCG')
user_dcg.head()
