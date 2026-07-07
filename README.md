# eunomia
Document clustering, visualization, and analysis of American state-level legislation.

## What is Eunomia?  
This project was inspired by political advocacy groups which produce model legislation to promote their regulatory vision across the country. The example bills these groups produce are regularly submitted at all levels of government - local, state, and federal - and not just once, but over and over, across numerous forums for years and years. It can be very difficult to fully assess the impact such groups actually have on the legislative regime that the average American is subject to.

## Who actually produces our legislation? 
The goal of this project is to help identify groups of bills which can be traced back to a common source, with a particular focus on bills introduced across multiple states. Eunomia applies natural language processing (NLP) techniques to perform forensic authorship, identifying the groups and individuals actually responsible for creating them.

Special interest groups from across the political spectrum have become ever more adept at crafting and promoting "model" legislation in service of their political agendas, essentially promoting these pre-written bills, with "INSERT STATE HERE" placeholders, to state lawmakers. The politicians, in turn, often submit the model bills in their state legislatures with only minor edits - if they even bother to modify the language at all.

The process of identifying what issues to legislate on, researching various regulatory schemes, and drafting legislation to promote a particular outcome can be time-consuming and expensive. It also requires a degree of legal expertise that most part-time lawmakers do not have. By providing lawmakers with model bills, special interest groups and legislators establish a symbiotic relationship. The interest groups make lawmakers' jobs easy by identifying legislatable issues for them, providing them with turnkey bills they can champion on the campaign trail; these groups' preferred policy agendas get promoted, and financial and political support are given to lawmakers in exchange.

Noticeably absent from this relationship is the electorate. Special interest groups may have little to no connection to the state(s) where their model legislation is enacted. Further, the texts of model bills are frequently negotiated behind closed doors, with no public oversight or input into this key step of the legislative process.

This raises very serious questions about the nature of American democracy: Who is actually producing the laws that govern us? Whose interests are being served by this system?

A lot of attention is paid to the sources of a candidate's campaign contributions as a metric of political influence and bias. The goal of Eunomia is to complement that analysis with a look at the influences on a legislator's material outputs once in office. "Whose homework are they copying?" is, or should be, just as pertinent as, "Whose money is supporting their campaigns?"

# Data model
Eunomia uses two-tiered clustering to identify groups of bills by topic (first tier clusters), and then within each general topic, bills are analyzed for fine-grained textual similarity (second tier clusters).

This approach aims to minimize the computational expense of applying unsupervised models to a large number of documents. By filtering out a significant number of common stopwords, truncating ("stemming") terms to their common roots, and limiting the number of terms used to produce tokenized n-grams, the basis clusters (first tier) paint with a relatively broad brush. The idea is that this should capture the general vocabulary of the texts, establishing broad clusters across a given sample. Then, the basis clusters of interest are re-clustered with fewer filters, allowing more detailed lexical features to produce sub-clusters using specific, identifying language that may point to shared authorship.

Topic-level clusters can also be used to identify general trends in legislation across states and over specific time spans.

## How is this different from a standard hierarchical clustering approach?
Off-the-shelf hierarchical clustering methods featurize the documents in a collection one time, and then use those features to build a comprehensive hierarchy of all available documents.

The standard hierarchical cluster analysis (HCA) methods are either bottom-up (agglomerative) or top-down (divisive). In bottom-up HCA, the two most similar documents would form the initial cluster, with the next-most-similar document then being added recursively until all documents form a single cluster. In the top-down approach, all documents start in a single cluster, which then gets recursively split into two or more clusters until a stopping metric is reached or all documents have been split into a "cluster" of n=1.

Eunomia could be thought of as a form of top-down clustering, although as described above it uses different document features for the first-pass vs second-pass clusters. It also is not exhaustive - after the first pass is complete, sub-clustering is performed on documents within each basis cluster separately; unclustered documents are excluded from further analysis.

# How to use it
The `eunomia` package can be installed via pip:
```
(TBD)
```

## Example runbooks and analysis
The pages in the [docs](docs) folder walks step by step through the details of the pipeline, with example usage and potential customizations.

# Why "Eunomia"?
<img align="right" width="200" src="imgs/eunomia.jpg" alt="Eunomia by Herman Rosse" />
<a href="https://en.wikipedia.org/wiki/Eunomia">Eunomia</a> was a minor Greek deity dedicated to good laws and good governance. May she look favorably upon our work.
<div style="clear: both;"></div>