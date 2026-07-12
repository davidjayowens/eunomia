# eunomia
Document clustering, visualization, and analysis of American state-level legislation.

## What is Eunomia?  
This project was inspired by political advocacy groups which draft model legislation to promote their regulatory agendas. The generic bills these groups produce can be introduced across multiple states, often with only minor adjustments from the source text. It can be very difficult to fully assess the impact such groups actually have on the legislative regimes that Americans are subject to.

The goal of Eunomia is to help identify bills which can be traced back to a common source. Eunomia uses a combination of natural language processing (NLP) and clustering techniques to conduct forensic authorship, identifying the people responsible for creating and sponsoring model texts.

# The Data model
Eunomia uses two-tiered clustering to identify groups of bills by topic (first-order clusters), and then within each general topic, bills are analyzed for fine-grained textual similarity (second-order clusters). This approach aims to minimize the computational expense of applying unsupervised models to a large number of documents. 

## Text cleaning
For first-order clusters, standard natural language processing techniques are applied to simplify texts and improve matchability, which facilitates baseline clustering. 

Although all parameters can be overridden by the user, the default is to filter out a customizable list of common terms ("stopwords"), truncate ("stem") all vocabulary to word roots, and then produce concatenated n-grams with a relatively low value of `n=3`. This should capture the general vocabulary of the texts, establishing broad clusters across a given sample. 

The filtered, stemmed n-gram vocabulary is then used to produce TF-IDF ("term frequency - inverse document frequency") vectors for the entire sample.

For second-order clusters, TF-IDF vectors are re-calculated on a per-cluster basis, skipping the filtering and stemming steps to preserve the vocabulary as-is. Additionally, a higher value of `n=7` is used for producing more distinct n-grams. In this way, the document vectors produce a relatively unique "fingerprint" that, when clustered, may point to shared authorship.

## Clustering
The actual clustering approach used is density-based, via cosine similarity of the document vectors. By default, Eunomia uses the `DBSCAN` algorithm from `scikit-learn`, although it is parameterized to use `HDBSCAN` or `OPTICS`. It can also accept other clustering models provided as an input, as long as they use `.fit()` and include a `.labels_` attribute.

## Visualizing & analyzing clusters
After featurizing and clustering documents in the sample, two forms of visualization are used to analyze clusters. For specific details of using the `eunomia.visualizer` package, see [the docs](<docs/02 - Clustering, visualization, and analysis.md>).

### PlotPolars
The `PlotPolars` class represents clusters as polar line plots. All documents in each cluster are represented based on the strength of a parameterized number of features. Optionally, you can display the top vocabulary terms associated with each feature. Clusters can be stacked on a single comprehensive plot or split into per-cluster subplots.

**First-order clusters, unified plot**
![polar base clusters](imgs/polars_base_plot.png)

**Second-order clusters, separate subplots**
![polar sub clusters](imgs/polars_sub_plot.png)

When visualizing second-order clusters, only the sub-clusters within a single base cluster can be visualized at a time.

### PlotDocs
The `PlotDocs` class represents the intersection of documents in a cluster. Each "document" image plotted shows roughly where segments of text align across all documents in a given cluster.

**First-order clusters**
![document base clusters](imgs/docs_base_plot.png)

**Second-order clusters**
![document sub clusters](imgs/docs_sub_plot.png)

For second-order clusters, the sub-clusters are shown as a collection within a particular base cluster.

## How does this differ from standard hierarchical clustering models?
Basic hierarchical clustering methods featurize the documents in a collection one time, and then use those features to build a comprehensive hierarchy of all available documents.

The standard hierarchical cluster analysis (HCA) methods are either bottom-up (agglomerative) or top-down (divisive). In bottom-up HCA, the two most similar documents would form the initial cluster, with the next-most similar document then being added recursively until all documents form a single cluster. In the top-down approach, all documents start in a single cluster, which then gets recursively split into two or more clusters until a stopping metric is reached or all documents have been split into a "cluster" of n=1.

Eunomia is a hybrid model and not "true" hierarchical clustering, although it can be thought of as a pseudo-divisive approach. It produces first- and second-order clusters (aka, topic/base clusters and lexical/sub-clusters, respectively) similar to the hierarchical model, but it uses different document features at each level. It also is not exhaustive - after each pass is complete, unclustered documents are excluded from further analysis.

# How to install
After cloning the repository, the `eunomia` package can be installed via pip:
```
pip install <eunomia folder location>
```

## Example runbooks and analysis
The pages in the [docs](docs) folder walk step by step through the details of the pipeline, with example usage and potential customizations.

# Why "Eunomia"?
<img align="right" width="200" src="imgs/eunomia.jpg" alt="Eunomia by Herman Rosse" />
<a href="https://en.wikipedia.org/wiki/Eunomia">Eunomia</a> was a minor Greek deity dedicated to good laws and good governance. May she look favorably upon this work.
<div style="clear: both;"></div>