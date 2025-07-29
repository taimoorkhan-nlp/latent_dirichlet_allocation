
# Discovering Themes in Text with Topic Modeling (Latent Dirichlet Allocation)

## Description
The method uncovers hidden themes as semantic structures that are frequently discussed in the documents to explore unfamiliar domains. It may also be used to identify features as topics for subsequent tasks. For example, applying the method to the hotel reviews corpus may result in topics representing food quality, menu, table service, pricing, etc. It calculates the co-occurrence frequencies among words to organize topics as ordered collections of words and documents as ordered collections of topics. As an unsupervised approach, the topics are unlabeled and are represented by their most probable words. The method reads input as a document per line and outputs two files: document-topic distribution and topic-word distribution.

## Use Cases
To explore the main topics discussed in political poll reviews of different user groups and analyze how the topics vary across these groups. The topics may include health care, immigration, the economy, etc., and are represented through their most prominent words. 

## Input Data
The input can be any text to explore. For demonstration purposes, we use BBC news article headlines as sample documents. Below are 10 example headlines taken from the dataset, which can be found in the file [data/input.csv](https://github.com/taimoorkhan-nlp/latent_dirichlet_allocation/blob/master/data/input.csv)

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

| Topic Name | Words and Probabilities                                                                                   |
|------------|----------------------------------------------------------------------------------------------------------|
| Topic 0    | ('deal', 0.039125056962437336)('profit', 0.03261506412342946)('profits', 0.026105071284421584)('Japanese', 0.019595078445413708)('takeover', 0.01308508560640583)('lifts', 0.01308508560640583)("India's", 0.01308508560640583)('high', 0.01308508560640583)('Parmalat', 0.01308508560640583)('China', 0.01308508560640583)                             |
| Topic 1    | ('economy', 0.04184945338068379)('hits', 0.03488614998955504)('fuel', 0.03488614998955504)('Yukos', 0.02792284659842629)('growth', 0.02792284659842629)('Japan', 0.02792284659842629)('German', 0.020959543207297537)('$280bn', 0.013996239816168788)('French', 0.013996239816168788)('prices', 0.013996239816168788)|
| Topic 2    | ('jobs', 0.024660229998155092)('firm', 0.024660229998155092)('gets', 0.024660229998155092)('India', 0.018510546706844596)('sales', 0.018510546706844596)('new', 0.018510546706844596)('oil', 0.018510546706844596)('BMW', 0.018510546706844596)('trade', 0.012360863415534098)('rise', 0.012360863415534098)|

The complete distribution is written to [data/output-data/topic-word-distribution.txt](https://github.com/taimoorkhan-nlp/latent_dirichlet_allocation/blob/master/data/output-data/topic-word-distribution.txt)

**Document-topic distribution:** Each document is assigned probabilities of representing a topic based on the topic association of its words. These probabilities indicate the extent to which the topics are discussed in these documents. For example, document 1 can be 45% topic 0, 45% topic 1, and 10% topic 2.

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
It is the vanilla implementation of the Latent Dirichlet Allocation, built from scratch. The libraries  

It is the vanilla implementation of the Latent Dirichlet Allocation technique built from scratch; therefore, only basic libraries, i.e., `numpy`, `pandas`, `random`, and `string`, are needed to read data and generate random numbers.
- Update [config.json](config.json) to read method configurations in JSON format and update as desired. 
- Setup the environment using [requirements.txt](https://github.com/taimoorkhan-nlp/latent_dirichlet_allocation/blob/master/requirements.txt) through command `pip install -r requirements.txt`
- Put your data in [data/input.csv](data/input.csv)
- Execute the notebook [LDA-collapsed-gibbs-sampling.ipynb](LDA-collapsed-gibbs-sampling.ipynb) to get results

## How to Use
- Put your data in [data/input.csv](https://github.com/taimoorkhan-nlp/latent_dirichlet_allocation/blob/master/data/input.csv)
- Execute the first notebook [prepare-data.ipynb](https://github.com/taimoorkhan-nlp/latent_dirichlet_allocation/blob/master/prepare-data.ipynb) to transform the data into integer encoding
- Execute the main notebook `[LDA-collapsed-gibbs-sampling.ipynb](https://github.com/taimoorkhan-nlp/latent_dirichlet_allocation/blob/master/LDA-collapsed-gibbs-sampling.ipynb) to get results

## Technical Details
This method represents the [Latent Dirichlet Allocation (LDA)](https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf?ref=http://githubhelp.com) topic modeling approach. It uses collapsed Gibbs sampling [(LDA with collapsed Gibbs sampling)](https://www.cs.cmu.edu/~wcohen/10-605/papers/fastlda.pdf) as the inference technique, which is an efficient extension of Gibbs sampling. The inference technique decides the most suitable topic for the sampled word, given the current state of the model, where the state of the model is determined by its document-topic distribution (having probabilities of each topic in each document) and topic-word distribution (having probabilities of each word in each topic). When a word switches its topic, i.e., its most suitable topic in the current state is different from its present topic (assigned in the previous iteration), it results in changing the state of the model. In each iteration, the inference technique samples each word to estimate its most suitable topic given the current state of the model. At the end of the iteration, the document-topic and topic-word probabilities are recomputed, and the model's state is updated. 

The model starts with random initialization, i.e., the words are assigned to topics at random to help generate the initial probabilities for representing the initial state of the model. In the initial iterations, the word switch their topics more rigorously, which reduces in the subsequent iterations as more and more words settle in their respective topics. Using the [Markov chain Monte Carlo (MCMC) approach](https://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo), the next state of the model is determined from the current state of the model, converging on a stable distribution. Allowing collapsed Gibbs sampling with enough iterations (in a few thousand), the words are expected to settle down in their respective topics. Although starting from any random state, the model converges to similar probability distributions; however, a better initial state helps to converge in fewer iterations. As an unsupervised method, the topics are not labeled but rather represented by their highest probability (top presentable) words. Each topic is an ordered list of words, where the words are organized by their probabilities with the topic (calculated based on the co-occurrence frequency of the word with all the other words in the topic) in decreasing order.  

The method has two hyperparameters, i.e., $\alpha$ as the Dirichlet prior for the document-topic distribution and $\beta$ as the Dirichlet prior for the topic-word distribution. This implementation uses the optimum values advised for the method. These Dirichlet priors control the bias or variance in the model's distribution. For example, when the $\alpha$ value is 1, the document-topic distribution is fully drawn from the input data without the prior affecting it. However, when its value is higher (generally in multiples of 10 as 100, 1000, ...), it introduces bias into the model. The high prior value undermines the probabilities computed from the data, and therefore, most topics have similar probabilities in a document. On the other hand, when the $\alpha$ is below 1 (i.e., 0.1, 0.001, ...), it increases variance, assigning more weight to the probabilities from within the data, resulting in only a few topics with higher probabilities in the document. Thus, $\alpha$ allows for controlling the number of topics represented in a document. The Dirichlet prior $\beta$ plays a similar role in the topic-word distribution.

Topic modeling is also a soft clustering approach (derived from the concept of soft sets in mathematics), as all words belong to all topics and all topics belong to all documents with varying probabilities. However, for the sake of clarity, only higher probabilities are considered. Thus, in general, words having a higher probability for a topic have lower probabilities for all other topics, except for **polysemous words**. They can be among the highest probability words for multiple topics, where each occurrence is surrounded by its contextually correlated words.  

Topic models are generally represented by a plate-notation diagram as shown below [source](https://images.prismic.io/rosetta-marketing-website/9ee5d938-d009-4fcd-88c8-7bd539391da1_image+%2848%29.png?auto=compress,format). The rectangles (also called plates in the diagram) represent loops, while the circles represent variables. The $\alpha$ and $\beta$ priors are consumed by $\Theta$ and $\Phi$ representing document-topic and topic-word distributions, respectively. *W* represents the sampled word, while *Z* is the topic assigned to it. *K*, *M*, and *N* as the loop stopping condition, represent the number of topics, the number of documents in the corpus, and the number of words in the topics, respectively. It can also be observed that among all variables, only *W* has a gray background, as it is the only known variable. 

$\Theta_{d, t} = \frac{\eta_{d, t}^{-} + \alpha}{\sum_{k}^{K} \eta_{d,k}^{-} + K\alpha}$ 

and

$\Phi_{t, w} = \frac{\eta_{t, w}^{-} + \beta}{\sum_{v}^{V} \eta_{t,v}^{-} + V\beta}$

$P(z = t | z_-, w, d, .) = \Theta_{d, t} \times \Phi_{t, w}$

Where, *w* is the sampled word from *d*^{th} document, whose probability is computed for topic *t* in $\Theta_{d, t}$. The denominator normalizes the probabilities, having *K* as the total number of topics. $\Phi_{t, w}$ computes the probability of word *w* for topic *t*, where *V* is the vocabulary size. $\eta$ represents frequency e.g., $\eta_{d,t}$ means the number of times topic *t* appears in document *d*. While *-* is for excluding the scores of the sampled word to avoid bias in favor of the present topic.


<img src="https://images.prismic.io/rosetta-marketing-website/9ee5d938-d009-4fcd-88c8-7bd539391da1_image+%2848%29.png?auto=compress,format" alt="Alt Text" width="300" height="200">

The Vanilla implementation offers higher transparency and thus more control over the internal decisions of the method. 

## Contact details
M. Taimoor Khan (<a href="mailto:taimoor.khan@gesis.org">taimoor.khan@gesis.org</a>)
