# LearnRotate2

This repo contains some experiments about data augmentation that I became curious about, specifically augmenting MNIST digits by adding rotated versions of the digits.

My initial curiosity was spiked by wondering if the data was good enough to tell the difference between 6's and 9's, when rotated. It was! This experiment is seperated into run_sixes_and_nines.

The learner had a tremendous amount of difficulty on a dataset consisting of digits all randomly rotated, so I decided to try training it on datasets with a small number of digits rotated randomly by 0-360 degrees and most only rotated by a smaller amount.

Then I used Gaussian process regression to see which ratio of hard to easy (from 0 hard examples to only hard examples) performed best. To my delight, it does perform better with ~10% of the data consisting of those hard examples. Just like people, a little bit of extra challenge helps learning (but not too much.)
