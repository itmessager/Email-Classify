#  Improving Email Classification With One Line of Code

 Bayes classification algorithm is a classification method of statistics, which is a kind of algorithm which uses probabilistic statistical knowledge to classify.
The algorithm can be applied to large database, and the method is simple, the classification accuracy is high and the speed is fast. In this paper, we change the value of the priori probability to make the system more sensitive to high-frequency vocabulary, so as to improve the accuracy of the distinction between spam, in addition, we also changed the traditional Bayesian calculation method, improve the calculation speed.

## Dependencies
+ Python 3;
+ numpy  1.15;
+ pandas 0.23;

## More Sensitive Bayes Algorithm



## Results

The corpus used in the experiment is derived from the spam database in the UCI machine learning database.<br/>
trainset:testset = 2000:500,Performance can be approximately reproduced.<br/>
   
    |                Index            |      traditional Bayes    |    More Sensitive and Faster Bayes     | 
    |          total accuracy         |           0.960           |               0.982                    |
    |        ham email precision      |           0.99287         |               0.98409                  |
    |         spam email precision    |           0.78481         |               0.96667                  |
    |          ham email recall       |           0.96092         |               0.99540                  |
    |         spam email recall       |           0.95385         |               0.89231                  |
    |          train time cost        |           0.02523s        |               2.90399s                 |