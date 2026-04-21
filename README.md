# eunomia
Document clustering, visualization, and analysis of American state-level legislation.

## Who actually produces our legislation? 
The goal of this project is to help identify groups of bills introduced across multiple states and/or sessions which can be traced back to a common source. 

Special interest groups from across the political spectrum have become ever more adept at crafting and promoting "model" legislation in service of their political agendas, essentially handing pre-written bills to state lawmakers. Those lawmakers, in turn, often submit the model bills in their state legislatures with only minor edits - if they even bother to modify the language at all. [Include link to bills submitted completely without alteration or with, eg, "insert state here"]

The process of identifying what issues to legislate on, researching various regulatory schemes, and drafting legislation to effect a particular outcome can be time-consuming and expensive. By providing lawmakers with turnkey bills - which often come with promises of financial support for representatives who agree to advocate for them - special interest groups and legislators establish a symbiotic relationship. The interest groups make lawmakers' jobs easy by identifying legislatable issues for them, providing ready-to-submit bills on those issues, and promising financial and political support in return for advancing their agenda.

Noticeably absent from this relationship: the electorate. Special interest groups may have little to no connection to the state(s) where their model legislation is enacted. Further, the texts of model bills are frequently negotiated behind closed doors, with no public oversight or input into this step of the legislative process.

This raises very serious questions about the nature of American democracy: Who is actually producing the laws that governs us? Whose interests are being served by this system?

A lot of attention is paid to the sources of a candidate's campaign contributions as a metric of political influence and bias. Project Eunomia seeks to complement that analysis with a look at the influences on a legislator's material outputs once in office. "Whose homework are they copying?" is, in our view, just as important as, "Who paid to get them elected?"


## Why "Project Eunomia"?
<img align="right" width="200" src="imgs/eunomia.jpg" alt="Eunomia by Herman Rosse" />
<a href="https://en.wikipedia.org/wiki/Eunomia">Eunomia</a> was a minor Greek deity dedicated to good laws and good governance. May she look favorably upon our work.
<div style="clear: both;"></div>

# Data model
Eunomia uses a two-tiered clustering model to identify groups of bills by topic (first tier clusters), and then within each general topic, bills are analyzed for fine-grained textual similarity (second tier clusters).

This two-phase approach aims to minimize the computational expense of applying unsupervised models to a large number of documents. It also enables different modes of analysis, as the topic-level clusters can be used to identify general trends in legislation across states and over specific periods.
