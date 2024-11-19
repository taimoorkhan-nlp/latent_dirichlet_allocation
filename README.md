
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
- sample putput data is given in `data/output-data/document-topic-distribution.txt` and `data/output-data/topic-word-distribution.txt`

## Contact details
M. Taimoor Khan (taimoor.khan@gesis.org)
