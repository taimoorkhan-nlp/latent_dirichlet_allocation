
## Latent Dirichlet allocation with collapsed gibbs sampling (Topic Modeling)

Description:
Topic modeling is a potent methodology tailored for social scientists engaged in uncovering thematic structures within vast textual datasets. This probabilistic generative model assumes that each document is a blend of various topics, with each topic itself being a mix of words. It aids in unveiling the latent topic structures present in the documents. For social scientists grappling with extensive text data – be it interviews, surveys, or written documents – LDA offers an automated approach to discerning underlying themes and patterns.

The application of LDA in social science research proves invaluable in various aspects. Researchers can employ this method to conduct topic analysis, systematically categorizing and understanding prevalent themes within their datasets. LDA facilitates an exploration of the dataset, allowing for the identification of dominant topics, distribution of words associated with each topic, and overall thematic trends. Social scientists benefit from the efficiency of this automated process, saving time and resources. Moreover, LDA ensures an unbiased exploration, objectively identifying topics without preconceived notions. As a result, researchers can delve into qualitative analysis with a focus on interpreting results and deriving meaningful insights from their data. 

This particular latent dirichlet allocation implementation use collapsed gibbs sampling as an inference technique that is an efficient version of gibbs sampling and is efficient than standard inference techniques. This is vanilla implemenatation of the method that only uses numpy and gives users full control over the code and internal decisions with a lot of transparency that comes with it. Almost anything can be tuned according to the nature of the project. The implementation is in the form of a class, therefore, modifying individual methods or adding new methods will not effect the existing working of the code. However, the initial set of functions that reads the data and setup the initial state of the model using markov chain monte carlo approach must be dealt with care, as modifying it may affect the loading of the data and initial state of the model. The model expects the input data to be tokenized, integer encoded and presented in a .txt file with word integer tokens seperated by tab and documents by new lines. Overall, LDA with Collapsed Gibbs Sampling stands as a robust tool seeking enhanced qualitative analysis to extract richer understanding from large textual datasets.

## Use case
Social scientist examining the dynamics of political poll reviews to gain nuanced insights into voter sentiments. Utilizing topic modeling, particularly Latent Dirichlet Allocation (LDA), this researcher aims to identify latent topics within a large dataset of online reviews related to political polls. By applying LDA, distinct themes such as candidate preferences, policy discussions, and public sentiment emerge. This approach enables a detailed examination of the diverse perspectives within the political discourse, providing a comprehensive understanding of the factors influencing public opinion during election seasons. The results can inform political analysts, campaigners, and policymakers, contributing to a deeper comprehension of voter concerns and preferences.

## Structure
The whole repo code is a class in a single .ipynb file having functions for specific steps. It reads integer encoded data from data/inputdata.txt file.

## Environment setup
As it is vanilla implementation of the latent dirichlet allocation technique with everything built from scratch. It only uses numpy for efficient operations of data in lists.

## Sample input and output
It takes integer encoded text as input with tokens separated by tab and documents by new line. Sample input file is in the repo data/inputdata.txt having 3 input documents and vocabulary size of 3 as well. The vocabulary integer encoding used are 0, 1 and 2.
As an output, it has two functions that allows to analysis the data after training the model. The wordsPerTopic( function presents a matrix of raw probabilistic distribution of words across topics. The topicsPerDocuments( function presents a matrix of raw proababilitic distribution of topics across documents. 

## Contact details
M. Taimoor Khan (taimoor.nlp@gmail.com)


