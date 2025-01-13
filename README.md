
# Discovering Themes in Text with Topic Modeling (Latent Dirichlet Allocation)

## Description:

This method helps uncover hidden themes within a collection of text documents, making it a valuable tool for exploring unfamiliar domains. For example, a social scientist analyzing public discussions on social media or academic papers on a particular topic can use this method to identify recurring themes or topics.

The method assumes that each document contains a mixture of topics and that each topic comprises a distinct set of words. By processing a text file with one document per line, the method generates two key outputs:

- The probability of each topic appearing in each document.
- The most representative words for each topic, along with their probabilities.

This approach is built on [latent dirichlet allocation (LDA)](https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf?ref=http://githubhelp.com), using a specialized technique called collapsed Gibbs sampling [(LDA with collapsed gibbs sampling)](https://www.cs.cmu.edu/~wcohen/10-605/papers/fastlda.pdf). This enhances efficiency, producing a balanced topic distribution while allowing users control over the model’s internal workings.
It uses Markov chain monti carlo approach to initialize the model with a random state. The method provides vanila implementation (using only basic packages for loading data e.g., numpy, json and random number generation) of Topic modeling with maximum control to customize its behavior. It gives user control over internal decisions with transparent way. The method is implemented as a class to easily extend its behavior. 

## Keywords
topic modeling, latent dirichlet allocation, LDA

## Use Case(s)
A social scientist wants to examine the dynamics of political poll reviews to gain nuanced insights into voter interests.

## Structure
- [data/](data/)
- [data/input.csv](data/input.csv) BBC article headings, using only first 100 for demo

*generated by prepare-data.ipynb*
- [data/integer-encoded-data.txt](data/integer-encoded-data.txt)
- [data/dictionary.json](data/dictionary.json)
- [data/revdictionary.json](data/revdictionary.json)

*scripts*
- [LDA-collapsed-gibbs-sampling.ipynb](LDA-collapsed-gibbs-sampling.ipynb)
- [LICENSE.txt](LICENSE.txt)
- *requirements.txt not needed*
  
## Environment setup
As it is vanilla implementation of the latent dirichlet allocation technique with everything built from scratch. 
- Setup the environment using [requirements.txt](requirements.txt) through command `pip install -r requirements.txt`
- Put your data in [data/input.csv](data/input.csv)
- Execute the first notebook [prepare-data.ipynb](prepare-data.ipynb) to transform the data into integer encoding
- Execute the main notebook `[LDA-collapsed-gibbs-sampling.ipynb](LDA-collapsed-gibbs-sampling.ipynb) to get results 

## Sample input and output
- sample input data is given in [data/input.csv](data/input.csv)

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
|High fuel prices hit BA's profits|
|Peugeot deal boosts Mitsubishi|
|Ad sales boost Time Warner profit|
|Yukos unit buyer faces loan claim|

- sample putput data is given in [data/output-data/document-topic-distribution.txt](data/output-data/document-topic-distribution.txt)
  
| Document   | Topic 0                 | Topic 1                 | Topic 2             |
|------------|-------------------------|-------------------------|---------------------|
| Document 0 | 0.125                   | 0.75                    | 0.125               |
| Document 1 | 0.125                   | 0.375                   | 0.5                 |
| Document 2 | 0.375                   | 0.125                   | 0.5                 |
| Document 3 | 0.5                     | 0.125                   | 0.375               |
| Document 4 | 0.2857142857142857      | 0.42857142857142855     | 0.2857142857142857  |
| Document 5 | 0.14285714285714285     | 0.14285714285714285     | 0.7142857142857143  |
| Document 6 | 0.125                   | 0.5                     | 0.375               |
| Document 7 | 0.125                   | 0.0.5                   | 0.375               |
| Document 8 | 0.25                    | 0.375                   | 0.375               |
| Document 9 | 0.42857142857142855     | 0.42857142857142855     | 0.14285714285714285 |



and [data/output-data/topic-word-distribution.txt](data/output-data/topic-word-distribution.txt)

| Topic Name | Words and Probabilities                                                                                   |
|------------|----------------------------------------------------------------------------------------------------------|
| Topic 0    | ('profits', 0.028118645256293383)('profit', 0.028118645256293383)('firm', 0.028118645256293383)('Japan', 0.028118645256293383)('Japanese', 0.021106514269686554)('2004', 0.014094383283079725)('prices', 0.014094383283079725)('Parmalat', 0.014094383283079725)('recession', 0.014094383283079725)('wine', 0.014094383283079725)                             |
| Topic 1    | ('hits', 0.03138901071361443)('jobs', 0.025123739114090594)('growth', 0.025123739114090594)('India', 0.018858467514566754)('new', 0.018858467514566754)('oil', 0.018858467514566754)('trade', 0.012593195915042914)('takeover', 0.012593195915042914)('lifts', 0.012593195915042914)('production', 0.012593195915042914)|
| Topic 2    | ('economy', 0.0381320982171182)('deal', 0.0381320982171182)('fuel', 0.0317873231393947)('Yukos', 0.02544254806167121)('gets', 0.02544254806167121)('German', 0.019097772983947717)('sales', 0.019097772983947717)('BMW', 0.019097772983947717)('rise', 0.012752997906224223)('$280bn', 0.012752997906224223)|

## Contact details
M. Taimoor Khan (taimoor.khan@gesis.org)
