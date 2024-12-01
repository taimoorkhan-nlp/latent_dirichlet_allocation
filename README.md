
# Topic Modeling (Latent Dirichlet Allocation with Collapsed Gibbs Sampling)

Description:
Topic modeling is very handy in exploring new domains to uncover the thematic structures within it. It assumes that each document is a blend of various topics, where each topic itself is a mix of words. The method reads input data from a text file (document per line), and produces topic probabilities per document and top words (along with their probabilities) per topic.

The method implements latent dirichlet allocation (LDA) in specific with collapsed gibbs sampling that coverges to a more balanced distribution more efficiently than Gibbs sampling inference [LDA with collapsed gibbs sampling](https://www.cs.cmu.edu/~wcohen/10-605/papers/fastlda.pdf). It uses Markov chain monti carlo approach to initialize the model with a random state. The method provides vanila implementation (using only basic packages for loading data e.g., numpy, json and random number generation) of Topic modeling with maximum control to customize its behavior. It gives user control over internal decisions with transparent way. The method is implemented as a class to easily extend its behavior. 

## Keywords
topic modeling, latent dirichlet allocation, LDA

## Use case
Social scientist examining the dynamics of political poll reviews to gain nuanced insights into voter interests.

## Structure
- `data/` 
- `data/input-data.txt`

*generated by prepare-data.ipynb*
- `data/input-data/integer-encoded-data`
- `data/input-data/dictionary.json`
- `data/input-data/revdictionary.json`

*scripts*
- `prepare-data.ipynb`
- `LDA-collapsed-gibbs-sampling.ipynb`
- `LICENSE.txt`
- `requirements.txt`

## Environment setup
As it is vanilla implementation of the latent dirichlet allocation technique with everything built from scratch. 
- Setup the environment using `requirements.txt` through command `pip install -r requirements.txt`
- `Put your data in /data/input-data.txt`
- Execute the first notebook `prepare-data.ipynb` to transform the data into integer encoding
- Execute the main notebook `LDA-collapsed-gibbs-sampling.ipynb` to get results 

## Sample input and output
- sample input data is given in `data/input-data.txt`

| Headlines and Summaries |
|--------------------------|
| Wall St. Bears Claw Back Into the Black (Reuters) Reuters |
| Short-sellers, Wall Street's dwindling band of ultra-cynics, are seeing green again. |
| Carlyle Looks Toward Commercial Aerospace (Reuters) Reuters |
| Private investment firm Carlyle Group, which has a reputation for making well-timed and occasionally controversial plays in the defense industry, has quietly placed its bets on another part of the market. |
| Oil and Economy Cloud Stocks' Outlook (Reuters) Reuters |
| Soaring crude prices plus worries about the economy and the outlook for earnings are expected to hang over the stock market next week during the depth of the summer doldrums. |
| Iraq Halts Oil Exports from Main Southern Pipeline (Reuters) Reuters |
| Authorities have halted oil export flows from the main pipeline in southern Iraq after intelligence showed a rebel militia could strike infrastructure, an oil official said on Saturday. |
| Oil prices soar to all-time record, posing new menace to US economy (AFP) AFP |
| Tearaway world oil prices, toppling records and straining wallets, present a new economic menace barely three months before the US presidential elections. |
| Stocks End Up, But Near Year Lows (Reuters) Reuters |
| Stocks ended slightly higher on Friday but stayed near lows for the year as oil prices surged past barrel. |


- sample putput data is given in `data/output-data/document-topic-distribution.txt`
  
| Document   | Topic 0                 | Topic 1                 |
|------------|-------------------------|-------------------------|
| Document 0 | 0.6666666666666666      | 0.3333333333333333      |
| Document 1 | 0.5714285714285714      | 0.42857142857142855     |
| Document 2 | 0.8333333333333334      | 0.16666666666666666     |
| Document 3 | 0.8333333333333334      | 0.16666666666666666     |
| Document 4 | 0.8571428571428571      | 0.14285714285714285     |
| Document 5 | 0.125                   | 0.875                   |
| Document 6 | 0.125                   | 0.875                   |
| Document 7 | 0.25                    | 0.75                    |
| Document 8 | 0.25                    | 0.75                    |
| Document 9 | 0.6                     | 0.4                     |
| Document 10| 0.5714285714285714      | 0.42857142857142855     |
| Document 11| 0.7142857142857143      | 0.2857142857142857      |
| Document 12| 0.8571428571428571      | 0.14285714285714285     |
| Document 13| 0.14285714285714285     | 0.8571428571428571      |
| Document 14| 0.875                   | 0.125                   |
| Document 15| 0.14285714285714285     | 0.8571428571428571      |
| Document 16| 0.8571428571428571      | 0.14285714285714285     |
| Document 17| 0.7142857142857143      | 0.2857142857142857      |
| Document 18| 0.42857142857142855     | 0.5714285714285714      |
| Document 19| 0.42857142857142855     | 0.5714285714285714      |
 

and `data/output-data/topic-word-distribution.txt`

| Topic Name | Words and Probabilities                                                                                   |
|------------|----------------------------------------------------------------------------------------------------------|
| Topic 0    | enjoy: 0.0165, cooking: 0.0165, pasta: 0.0165, baking: 0.0165, dinner: 0.0015                             |
| Topic 1    | cooking: 0.0191, dinner: 0.0191, baking: 0.0191, enjoy: 0.0017, pasta: 0.0017                             |


## Contact details
M. Taimoor Khan (taimoor.khan@gesis.org)
