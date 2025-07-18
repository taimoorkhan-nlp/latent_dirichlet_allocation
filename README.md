
# Discovering Themes in Text with Topic Modeling (Latent Dirichlet Allocation)

## Description
This method helps uncover hidden themes within a collection of text documents, making it a valuable tool for exploring unfamiliar domains. For example, a social scientist analyzing public discussions on social media or academic papers on a particular topic can use this method to identify recurring themes or topics. The method assumes that each document contains a mixture of topics and each topic comprises a distinct set of words. By processing a text file with one document per line, the method generates two key outputs i.e., i) the probability of each topic appearing in each document ii) the most representative words for each topic, along with their probabilities.

This approach is built on [Latent Dirichlet Allocation (LDA)](https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf?ref=http://githubhelp.com), using a specialized technique called collapsed Gibbs sampling [(LDA with collapsed Gibbs sampling)](https://www.cs.cmu.edu/~wcohen/10-605/papers/fastlda.pdf). This enhances efficiency, producing a balanced topic distribution while allowing users control over the model’s internal workings.
It uses [Markov chain Monti Carlo approach](https://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo) to initialize the model with a random state. The method provides Vanilla implementation (using only basic packages for loading data e.g., numpy, JSON, and random number generation) of Topic modeling with maximum control to customize its behavior. It gives users transparent control over internal decisions. The method is implemented as a class to extend its behavior easily. 

## Use Case(s)
A social scientist wants to examine the dynamics of political poll reviews to gain nuanced insights into voter interests.

## Input Data
The input can be any text to explore. For demonstration purposes, we use BBC news article headlines as sample documents. Below are 10 example headlines taken from the dataset, which can be found in the file [data/input.csv](https://github.com/taimoorkhan-nlp/latent_dirichlet_allocation/blob/master/data/input.csv)

| Headlines |
|--------------------------|
| India calls for fair trade rules|
|Sluggish economy hits German jobs|
|Indonesians face fuel price rise|
|Court rejects $280bn tobacco case|
|Dollar gains on Greenspan speech|
|Mixed signals from French economy|
|Ask Jeeves tips online ad revival|
|Rank 'set to sell off film unit'|
|US trade gap hits record in 2004|
|India widens access to telecoms|

## Output Data

The latent topics identified are represented by the most significant words and their probabilities. It is similar to clustering in the sense that the words are grouped as topics and labeled unintuitively as topic 0, topic 1, etc. However, unlike clustering, the words have probabilities of relevance to the topic. Using these probabilities, only the top few words (10 in config.json) are used to represent a topic i.e., topic-word distribution.
For the three topics:

| Topic Name | Words and Probabilities                                                                                   |
|------------|----------------------------------------------------------------------------------------------------------|
| Topic 0    | ('deal', 0.039125056962437336)('profit', 0.03261506412342946)('profits', 0.026105071284421584)('Japanese', 0.019595078445413708)('takeover', 0.01308508560640583)('lifts', 0.01308508560640583)("India's", 0.01308508560640583)('high', 0.01308508560640583)('Parmalat', 0.01308508560640583)('China', 0.01308508560640583)                             |
| Topic 1    | ('economy', 0.04184945338068379)('hits', 0.03488614998955504)('fuel', 0.03488614998955504)('Yukos', 0.02792284659842629)('growth', 0.02792284659842629)('Japan', 0.02792284659842629)('German', 0.020959543207297537)('$280bn', 0.013996239816168788)('French', 0.013996239816168788)('prices', 0.013996239816168788)|
| Topic 2    | ('jobs', 0.024660229998155092)('firm', 0.024660229998155092)('gets', 0.024660229998155092)('India', 0.018510546706844596)('sales', 0.018510546706844596)('new', 0.018510546706844596)('oil', 0.018510546706844596)('BMW', 0.018510546706844596)('trade', 0.012360863415534098)('rise', 0.012360863415534098)|

The complete distribution is written to [data/output-data/topic-word-distribution.txt](https://github.com/taimoorkhan-nlp/latent_dirichlet_allocation/blob/master/data/output-data/topic-word-distribution.txt)

Topic Distribution Per Document: Each document is assigned probabilities of representing a topic based on the topic association of its words. These probabilities indicate the extent to which each document relates to specific topics. For example, document 0 can be 45% topic 0, 45% topic 1, and 10% topic 2.

In case a reader is interested in only reading more about topic 0, he/she may only focus on the documents where topic 0 is the major topic.
 
| Document   | Topic 0                 | Topic 1                 | Topic 2             |
|------------|-------------------------|-------------------------|---------------------|
| Document 0 | 0.125                   | 0.5                    | 0.375                |
| Document 1 | 0.375                   | 0.25                   | 0.375                |
| Document 2 | 0.125                   | 0.75                   | 0.125                |
| Document 3 | 0.375                   | 0.25                   | 0.375               |
| Document 4 | 0.7142857142857143      | 0.14285714285714285     | 0.14285714285714285  |
| Document 5 | 0.42857142857142855     | 0.2857142857142857     | 0.2857142857142857  |
| Document 6 | 0.125                   | 0.375                     | 0.5               |
| Document 7 | 0.25                   | 0.0.5                   | 0.25               |
| Document 8 | 0.375                    | 0.375                   | 0.25               |
| Document 9 | 0.14285714285714285     | 0.14285714285714285     | 0.7142857142857143 |
| ...|

Written in file [data/output-data/document-topic-distribution.txt](https://github.com/taimoorkhan-nlp/latent_dirichlet_allocation/blob/master/data/output-data/document-topic-distribution.txt)

## Hardware Requirements
The method runs on a small virtual machine provided by a cloud computing company (2 x86 CPU core, 4 GB RAM, 40GB HDD).

## Environment setup
It is the vanilla implementation of the Latent Dirichlet Allocation technique with everything built from scratch, therefore only basic libraries i.e., `numpy`, `pandas`, `random` and `string` are needed to read data and generate random numbers.
- Update [config.json](config.json) to read method configurations in JSON format and update as desired. 
- Setup the environment using [requirements.txt](https://github.com/taimoorkhan-nlp/latent_dirichlet_allocation/blob/master/requirements.txt) through command `pip install -r requirements.txt`
- Put your data in [data/input.csv](data/input.csv)
- Execute the notebook [LDA-collapsed-gibbs-sampling.ipynb](LDA-collapsed-gibbs-sampling.ipynb) to get results

## How to Use
- Put your data in [data/input.csv](https://github.com/taimoorkhan-nlp/latent_dirichlet_allocation/blob/master/data/input.csv)
- Execute the first notebook [prepare-data.ipynb](https://github.com/taimoorkhan-nlp/latent_dirichlet_allocation/blob/master/prepare-data.ipynb) to transform the data into integer encoding
- Execute the main notebook `[LDA-collapsed-gibbs-sampling.ipynb](https://github.com/taimoorkhan-nlp/latent_dirichlet_allocation/blob/master/LDA-collapsed-gibbs-sampling.ipynb) to get results

## Contact details
M. Taimoor Khan (<a href="mailto:taimoor.khan@gesis.org">taimoor.khan@gesis.org</a>)
