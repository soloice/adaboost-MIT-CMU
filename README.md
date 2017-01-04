# adaboost-MIT-CMU

Date: 2016, May.
Author: Soloice.

In this repo, Adaboost algorithm is implemented and tested with the MIT-CMU data set.

Adaboost is a well-known algorithm which is able to boost weak classifers into strong ones.  For human face recognition, a classical classifier is based on Haar-like features and combine these weak features to make a final strong classifier.

In this project, 26158 Haar-like features are chosen and it takes about half an hour to run over 6977 images of size 19-by-19 in the training set and calculate all these features.  For the boost procedure, in each iteration, 300 features out of the whole 29158 features are randomly picked and choose a best one to form a weak classifier.  Then the current weak classifier is intergrated into the final strong one.  After 80 iterations,  a strong classifier composed of 80 weak ones are generated, and achieves an accuracy of more than 99.9% on the training set, which agrees with the exponentially convergence rate of Adaboost algorithm. It takes about 10s to perform an iteration.

Though, the final strong classifier doesn't perform very well on the testing set. The confusing matrix is as follows:

| - | predicted negative | predicted positive |
|:-----:|:-----:|:-----:|
| true negative | 23427 | 146 |
|true positive | 380 | 92 |

This might be due to the different distribution of data in training and testing set: In the training set, there are 2429 positive samples and 4548 negative ones, which is roughly balanced; But for the testing set, the number of negative examples (23573) are far larger than that of positive ones (472).
