
# Discovering Themes in Text with Topic Modeling (Latent Dirichlet Allocation)

## Description
The method uncovers frequently discussed topics (or themes) in a document collection. The topics are represented by their most prominent words. These topics allow to explore document collections or can be used as labels in subsequent tasks. For example, applying the method to a hotel reviews corpus may result in topics that represent room quality, food quality, pricing, and so on, with each review being linked to the respective topics. In technical terms, it calculates the co-occurrence frequencies of words in documents to organize documents as distribution of topics and topics as distribution of words. The method reads input as a document per line and outputs two files, containing the document-topic distribution and topic-word distribution respectively.

This approach is built on [Latent Dirichlet Allocation (LDA)](https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf?ref=http://githubhelp.com), using a specialized technique called collapsed Gibbs sampling [(LDA with collapsed Gibbs sampling)](https://www.cs.cmu.edu/~wcohen/10-605/papers/fastlda.pdf). This enhances efficiency, producing a balanced topic distribution while allowing users control over the model’s internal workings.

It uses a [Markov chain Monte Carlo approach](https://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo) to initialize the model with a random state. The method provides a vanilla implementation (using only basic packages for loading data e.g., numpy, JSON, and random number generation) of topic modeling with maximum control to customize its behavior. It gives users transparent control over internal decisions. The method is implemented as a class to extend its behavior easily.

## Use Case(s)
This method is useful for anyone who needs to identify recurring themes or topics in large collections of text. For example:

- A social scientist analyzing public discussions on social media or academic papers to identify recurring themes or topics.
- A researcher examining the dynamics of political poll reviews to gain nuanced insights into voter interests.

By applying this method, users can explore unfamiliar domains and discover hidden patterns in their text data.

## Input Data
The input can be any text to explore. For demonstration purposes, we use BBC news article headlines as sample documents. Below are 10 example headlines taken from the dataset, which can be found in the file [data/input-data/news-headings.txt](data/input-data/news-headings.txt)

| Headlines |
|--------------------------|
| India calls for fair trade rules |
| Sluggish economy hits German jobs |
| Indonesians face fuel price rise |
| Court rejects $280bn tobacco case |
| Dollar gains on Greenspan speech |
| Mixed signals from French economy |
| Ask Jeeves tips online ad revival |
| Rank 'set to sell off film unit' |
| US trade gap hits record in 2004 |
| India widens access to telecoms |
|...|

## Output Data

The latent topics identified are represented by the most significant words and their probabilities. It is similar to clustering in the sense that the words are grouped as topics and labeled un-intuitively as topic 1, topic 2, etc. However, unlike clustering, the words have probabilities of relevance to the topic. 

**Topic-word distribution:** Topic word distribution provides the top few words for each topic with their probabilities, organized in decreasing order of their probabilities, indicating relevance to the topic. 

The following are 3 topics (`numTopics=3` in `config.json`) from the sample data, each with its top 10 words (`wordsPerTopic=10` in `config.json`) and their probabilities.

|     | Topic 1 | Topic 2 | Topic 3 |
|---------|---------|---------|---------|
|	w1 | (new, 0.0158)	|(win, 0.009)	| (film, 0.0123) |
|	w2 | (Blair, 0.0136)|	(deal, 0.0087) |	(set, 0.0089) |
|	w3 | (hits, 0.0096)	| (show, 0.0087)	| (top, 0.008) |
|	w4 | (net, 0.009)	| (shares, 0.0068)	| (hit, 0.0077) |
|	w5 | (election, 0.0071)|	(plan, 0.0068)	| (wins, 0.0074) |
|	w6 | (Labour, 0.0071)|	(firm, 0.0065)	| (return, 0.0071) |
|	w7 | (growth, 0.0065)|	(China, 0.0065)	| (bid, 0.0071) |
|	w8 | (face, 0.0062)	|( back, 0.0065)	| (gets, 0.0065) |
|	w9 | (says, 0.0062)	| (takes, 0.0065)	| (Brown, 0.0065) |
|	w10 | (row, 0.0062)	| (Yukos, 0.0062)	| (economy, 0.0061) |



The complete distribution is written to [data/output-data/topic-word-distribution.tsv](data/output-data/topic-word-distribution.tsv)

**Document-topic distribution:** Each document is assigned probabilities of representing a topic based on the topic association of its words. These probabilities indicate the extent to which the topics are discussed in these documents. For example, document 1 can be 37.5% topic 1, 25% topic 2, and 37.5% topic 3.

In case a reader is interested in only reading more about topic 1, he/she may only focus on the documents where topic 1 is the major topic.
 
|   |Topic 1   | Topic 2    | Topic 3  | Text                |
|-----------|------------|----------|----------|---------------------|
|	Doc 1   | 0.3750	| 0.2500	 | 0.3750	| India calls for fair trade rules |
|	Doc 2   | 0.3750	| 0.2500	 | 0.3750	| Sluggish economy hits German jobs |
|	Doc 3   | 0.3750	| 0.2500	 | 0.3750	| Indonesians face fuel price rise  |
|	Doc 4   | 0.3750	| 0.3750	 | 0.2500	| Court rejects $280bn tobacco case |
|	Doc 5   | 0.1429	| 0.2857	 | 0.5714	| Dollar gains on Greenspan speech |
| ...       | 

Written in file [data/output-data/document-topic-distribution.tsv](data/output-data/document-topic-distribution.tsv)

## Hardware Requirements
The method runs on a small virtual machine provided by a cloud computing company (2 x86 CPU core, 4 GB RAM, 40GB HDD).

## Environment setup
To set up the working environment, execute the command

`pip install -r requirements.txt`

## How to Use
- Put your data in [data/input.csv](https://github.com/taimoorkhan-nlp/latent_dirichlet_allocation/blob/master/data/input.csv)
- Execute the first notebook [prepare-data.ipynb](https://github.com/taimoorkhan-nlp/latent_dirichlet_allocation/blob/master/prepare-data.ipynb) to transform the data into integer encoding
- Execute the main notebook `[LDA-collapsed-gibbs-sampling.ipynb](https://github.com/taimoorkhan-nlp/latent_dirichlet_allocation/blob/master/LDA-collapsed-gibbs-sampling.ipynb) to get results

## Technical Details

 This method represents the [Latent Dirichlet Allocation (LDA)](https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf?ref=http://githubhelp.com) topic modeling approach. It uses collapsed Gibbs sampling [(LDA with collapsed Gibbs sampling)](https://www.cs.cmu.edu/~wcohen/10-605/papers/fastlda.pdf) as the inference technique, which is an efficient extension of Gibbs sampling. The inference technique decides the most suitable topic for the sampled word, given the current state of the model, where the state of the model is determined by its document-topic distribution (having probabilities of each topic in each document) and topic-word distribution (having probabilities of each word in each topic). When a word switches its topic, i.e., its most suitable topic in the current state is different from its present topic (assigned in the previous iteration), it results in changing the state of the model. In each iteration, the inference technique samples each word to estimate its most suitable topic given the current state of the model. At the end of the iteration, the document-topic and topic-word probabilities are recomputed, and the model's state is updated. 



The model starts with random initialization, i.e., the words are assigned to topics at random to help generate the initial probabilities for representing the initial state of the model. In the initial iterations, the word switch their topics more rigorously, which reduces in the subsequent iterations as more and more words settle in their respective topics. Using the [Markov chain Monte Carlo (MCMC) approach](https://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo), the next state of the model is determined from the current state of the model, converging on a stable distribution. Allowing collapsed Gibbs sampling with enough iterations (in a few thousand), the words are expected to settle down in their respective topics. Although starting from any random state, the model converges to similar probability distributions; however, a better initial state helps to converge in fewer iterations. As an unsupervised method, the topics are not labeled but rather represented by their highest probability (top presentable) words. Each topic is an ordered list of words, where the words are organized by their probabilities with the topic (calculated based on the co-occurrence frequency of the word with all the other words in the topic) in decreasing order.  
 @@ -106,4 +97,5 @@ Where, *w* is the sampled word from *d*^{th} document, whose probability is comp
The Vanilla implementation offers higher transparency and thus more control over the internal decisions of the method. 

## Contact details
M. Taimoor Khan (<a href="mailto:taimoor.khan@gesis.org">taimoor.khan@gesis.org</a>)
