from __future__ import annotations

from typing import Literal
import pandas as pd

from sklearn.cluster import DBSCAN, HDBSCAN, OPTICS
from scipy.cluster import hierarchy

from featurizer import make_bow, make_gram_tf, make_df, make_vocab, make_tfidf


class DocCluster:
    def __init__(self,
                 df: pd.DataFrame,
                 text_col: str,
                 verbose: bool = False):
        """
        Parameters
        ----------
        df : pandas DataFrame
            The data to use for clustering.

        text_col : str
            The label of the column containing 

        verbose : bool, default True
            If True, print messages while processing.
        """
        self._df = df.copy()
        self._col = text_col

        self.verbose = verbose

    # END OF __init__


    def __repr__(self):
        return(f"DocCluster(df=<pd.DataFrame>, text_col={self.col}, verbose={self.verbose})")


    @property
    def df(self):
        return(self._df)
    @df.setter
    def df(self, new_df):
        self._df = new_df.copy()

    @property
    def col(self):
        return(self._col)
    @df.setter
    def col(self, new_col):
        self._col = new_col


    def _make_features(self,
                       level: int,
                       use_level: int,
                       use_clusters: list[int] | None = None,
                       force: bool = False,
                       filter_bow: bool = True,
                       stem_bow: bool = False,
                       gram_n: int = 3,
                       min_df: float = 0.005,
                       max_df: float = 0.25,
                       tfidf_norm: bool | Literal['max'] = False):
        """
        Create bag of words (bow), stem-/n-grams, and term frequency (tf) 
        vectors for a set of bills; these will be used by _make_clusters().

        level : int
            Feature level.
                > 1 = First-order cluster featurization
                > 2 = Second-order cluster featurization
                > ...

        use_level : int
            Feature level of term frequencies to use as basis.
        
        use_clusters : list of int, optional
            If provided, limit featurization to bills in only these clusters
            from the specified level (use_level=).

        force : bool, default False
            If True, always produce features from scratch;
            if False, only produces bag of words and term frequencies if 
            they do not already exist in the sample features.

        filter_bow : bool, default True
            If True, apply filters (numbers, stopwords, etc) to bag of words;
            if False, retain all text tokens unmodified.

        stem_bow : bool, default False
            If True, apply stemming to bag of words;
            if False, retain all text tokens unmodified.

        gram_n : int, default 3
            Maximum n-gram length for bag of words featurization;
            term frequencies will include all n-grams of length 1 to gram_n.
        """
        # Labels for new features
        tf_col = f'lvl{level}_tf'
        tfidf_col = f'lvl{level}_tfidf'
        # Labels for basis features
        use_tf_col = f'lvl{use_level}_tf'
        use_cluster_id_col = f'lvl{use_level}_cluster'
        
        if self.verbose:
            print(f"Creating new features with parameters:\n"
                  f"General parameters: {level=} // {force=}\n"
                  f"Bag of Words: {filter_bow=} // {stem_bow=}\n"
                  f"n-gram Size: {gram_n=}\n"
                  f"Vector Vocab: {min_df=} // {max_df=}\n"
                  f"TF-IDF normalization: {tfidf_norm}\n"
                  f"\nFeature labels: {tf_col}, {tfidf_col}\n")

        if level == 1:
            # Make BoW
            if ('bow' not in self.df.columns) or force:
                if self.verbose:
                    print("Making Bag of Words...")
                self.df['bow'] = self.df.apply(lambda x: make_bow(text=x[self.col], filter=filter_bow, stem=stem_bow), axis=1)
                if self.verbose:
                    print("Bag of Words complete.\n")

            # Make term frequency vectors
            if (tf_col not in self.df.columns) or force:
                if self.verbose:
                    print(f"Making Level {level} Term Frequencies...")
                self.df[tf_col] = None
                for idx in self.df.index:
                    self.df.at[idx, tf_col] = make_gram_tf(bow=self.df.at[idx, 'bow'], gram_n=gram_n)
                if self.verbose:
                    print("TF vectors complete.\n")

            # Final features
            if (tfidf_col not in self.df.columns) or force:
                # Make document frequency vector
                if self.verbose:
                    print(f"Making Level {level} Document Frequencies...")
                df_vector = make_df(tf_vectors=self.df[tf_col])
                if self.verbose:
                    print("DF vectors complete.\n")

                # Make final vector vocabulary
                if self.verbose:
                    print(f"Making Level {level} Vector Vocab...")
                vector_vocab = make_vocab(df=df_vector, min_df=min_df, max_df=max_df)
                if self.verbose:
                    print(f"Vector Vocab complete - contains {len(vector_vocab)} terms.\n")

                # Make TF-IDF vectors
                if self.verbose:
                    print(f"Making Level {level} Topic Frequency - Inverse Document Frequency vectors...")
                self.df[tfidf_col] = None
                for idx in self.df.index:
                    self.df.at[idx, tfidf_col] = make_tfidf(tf=self.df.at[idx, tf_col], df=df_vector, vocab=vector_vocab, norm=tfidf_norm)
                if self.verbose:
                    print("TF-IDF vectors complete.\n")

        else: # Sub-clustering
            if use_clusters is None:     # Use all available clusters
                use_clusters = self.df[use_cluster_id_col].unique().tolist()

            # Sample-wide features
            if force:
                # Make BoW
                if self.verbose:
                    print("Remaking Bag of Words...")
                self.df['bow'] = self.df.apply(lambda x: make_bow(text=x[self.col], filter=filter_bow, stem=stem_bow), axis=1)
                if self.verbose:
                    print("Bag of Words complete.\n")

                # Make term frequency vectors
                if self.verbose:
                    print(f"Making Level {level} Term Frequencies...")
                self.df[tf_col] = None
                for idx in self.df.index:
                    self.df.at[idx, tf_col] = make_gram_tf(bow=self.df.at[idx, 'bow'], gram_n=gram_n)
                if self.verbose:
                    print("TF vectors complete.\n")

                # Use new TF vector for remaining features
                use_tf_col = tf_col

                # Create placeholder for TF-IDF vectors
                self.df[tfidf_col] = None

            # Sub-cluster featurization
            for cluster_id in use_clusters:
                # Get cluster rows
                this_cluster_idxs = self.df.loc[self.df[use_cluster_id_col]==cluster_id].index
                this_cluster_size = len(this_cluster_idxs)

                if self.verbose:
                    print(f"\n========== Level {level}: Cluster {use_level}-{cluster_id} // {this_cluster_size} documents ==========\n")

                # Make document frequency vector & vector vocabulary
                if self.verbose:
                    print(f"Making Level {level} Document Frequencies for Cluster {use_level}-{cluster_id}...")
                df_vector = make_df(tf_vectors=self.df.loc[this_cluster_idxs, use_tf_col])
                if self.verbose:
                    print("DF vectors complete.")

                if self.verbose:
                    print(f"Making Level {level} Vector Vocab for Cluster {use_level}-{cluster_id}...")
                vector_vocab = make_vocab(df=df_vector, min_df=min_df, max_df=max_df)
                if self.verbose:
                    print(f"Vector Vocab complete - contains {len(vector_vocab)} terms.")

                # Make TF-IDF vectors
                if self.verbose:
                    print(f"Making Level {level} Topic Frequency - Inverse Document Frequency vectors for Cluster {use_level}-{cluster_id}...")
                for idx in this_cluster_idxs:
                    self.df.at[idx, tfidf_col] = make_tfidf(tf=self.df.at[idx, use_tf_col], df=df_vector, vocab=vector_vocab, norm=tfidf_norm)
                if self.verbose:
                    print("TF-IDF vectors complete.\n")


    def _make_clusters(self,
                       level: int,
                       use_level: int,
                       use_clusters: list[int] | None = None,
                       drop_unclustered: bool = False,
                       cluster_method: Literal['dbscan', 'hdbscan', 'optics', 'hierarchy'] = 'dbscan',
                       **cluster_params):
        """
        Calculate bills' TF-IDF vectors and cluster them by similarity.

        level : int
            Feature level.
                > 1 = First-order clustering
                > 2 = Second-order clustering
                > ...
        
        drop_unclustered : bool, default False
            If True, after clustering, drop rows that were unclustered (cluster ID = -1)
        
        """
        clustering_methods = {
            'dbscan': DBSCAN,
            'hdbscan': HDBSCAN,
            'optics': OPTICS,
            'hierarchy': hierarchy
        }
        this_method = cluster_method.strip().lower()
        clusterer = clustering_methods.get(this_method)
        if clusterer is None:
            raise ValueError(f"Invalid parameter: {cluster_method=}\nMust be one of: {list(clustering_methods.keys())}")

        """
        DBSCAN(eps=0.5, *, min_samples=5, metric='euclidean', metric_params=None, algorithm='auto', leaf_size=30, p=None, n_jobs=None)
        HDBSCAN(min_cluster_size=5, min_samples=None, cluster_selection_epsilon=0.0, max_cluster_size=None, metric='euclidean', metric_params=None, alpha=1.0, algorithm='auto', leaf_size=40, n_jobs=None, cluster_selection_method='eom', allow_single_cluster=False, store_centers=None, copy='warn')
        OPTICS(*, min_samples=5, max_eps=inf, metric='minkowski', p=2, metric_params=None, cluster_method='xi', eps=None, xi=0.05, predecessor_correction=True, min_cluster_size=None, algorithm='auto', leaf_size=30, memory=None, n_jobs=None)
        """
        # Default clustering parameters
        if cluster_params.get('min_samples') is None:
            if level == 1:
                cluster_params['min_samples'] = 3
            else:
                cluster_params['min_samples'] = 2
        if cluster_params.get('metric') is None:
            cluster_params['metric'] = 'cosine'
        if cluster_params.get('n_jobs') is None:
            cluster_params['n_jobs'] = -1
        # Model-specific parameters
        if this_method == 'dbscan':
            if cluster_params.get('eps') is None:
                if level == 1:
                    cluster_params['eps'] = 0.75
                else:
                    cluster_params['eps'] = 0.01
        if this_method == 'hdbscan':
            if cluster_params.get('min_cluster_size') is None:
                cluster_params['min_cluster_size'] = cluster_params['min_samples']
        if this_method == 'optics':
            pass
        
        # Labels for new features
        cluster_id_col = f'lvl{level}_cluster'
        cluster_lbl_col = f'lvl{level}_cluster_vocab'
        # Labels for basis features
        use_tfidf_col = f'lvl{use_level}_tfidf'
        use_cluster_id_col = f'lvl{use_level}_cluster'

        if level == 1:
            # Make clusters
            if self.verbose:
                print(f"Making Level {level} Clusters...")
            # Rearrange the TF-IDF vectors into their own DataFrame
            tfidf_df = pd.DataFrame.from_dict(dict(zip(self.df[use_tfidf_col].index, self.df[use_tfidf_col].values))).T.fillna(0)
            clusters = clusterer(**cluster_params).fit(tfidf_df)
            self.df[cluster_id_col] = clusters.labels_ 
            if self.verbose:
                print(f"Clusters complete.\n")

            # Drop unclustered bills from the sample
            if drop_unclustered:
                if self.verbose:
                    print(f"Dropping unclustered bills from the sample...")
                samp_size = self.df.shape[0]
                self.df = self.df.loc[self.df[cluster_id_col] != -1].copy()
                samp_size2 = self.df.shape[0]
                if self.verbose:
                    print(f"{samp_size - samp_size2} rows dropped from the sample.\n")

            # Create "topic" label for each cluster ID
            # schema: "{top avg tf-idf score}-[{top 20 individual terms by avg tf-idf score}]"
            if self.verbose:
                print(f"Creating term labels for each cluster...")
            # FIXME
            for cluster_id in self.df[cluster_id_col].unique():
                #top20_1grams = tfidf_df.loc[self.df[cluster_id_col].loc[self.df[cluster_id_col] == cluster_id].index, 
                top20_1grams = tfidf_df.loc[self.df.loc[self.df[cluster_id_col] == cluster_id].index, 
                                            [col for col in tfidf_df.columns if '-' not in col]]\
                                            .mean().sort_values(ascending=False)[:20]
                cluster_lbl = f'{top20_1grams.iloc[0]:.4f}-' + '-'.join(top20_1grams.index)
                self.df.loc[self.df[cluster_id_col] == cluster_id, cluster_lbl_col] = cluster_lbl
            if self.verbose:
                print(f"Term labels complete.\n")
            
        else:   # SUB-CLUSTERING
            if use_clusters is None:     # Use all available clusters
                use_clusters = self.df[use_cluster_id_col].unique().tolist()
                
            for cluster_id in use_clusters:
                # Get basis cluster rows
                this_cluster_idxs = self.df.loc[self.df[use_cluster_id_col]==cluster_id].index
                this_cluster_tfidf = self.df.loc[this_cluster_idxs, use_tfidf_col]
                this_cluster_size = len(this_cluster_idxs)

                if self.verbose:
                    print(f"\n========== Level {level}: Cluster {use_level}-{cluster_id} // {this_cluster_size} documents ==========\n")

                # Make sub-clusters
                if self.verbose:
                    print(f"Making Level {level} Clusters for Cluster {use_level}-{cluster_id}...")
                # Rearrange the TF-IDF vectors into their own DataFrame
                tfidf_df = pd.DataFrame.from_dict(dict(zip(this_cluster_tfidf.index, this_cluster_tfidf.values))).T.fillna(0)
                # Make sub-clusters
                clusters = clusterer(**cluster_params).fit(tfidf_df)
                self.df.loc[this_cluster_idxs, cluster_id_col] = clusters.labels_ 
                # Backfill unclustered rows with -1
                self.df[cluster_id_col].fillna(-1, inplace=True)
                if self.verbose:
                    print(f"Clusters complete.")

                # Create "topic" label for each cluster ID
                # FIXME
                # schema: "{top avg tf-idf score}-[{top 20 individual terms by avg tf-idf score}]"
                if self.verbose:
                    print(f"Creating term labels for each cluster...")
                #self.df.loc[cluster_lbl_col] = ''
                #for sub_cluster_id in self.df.loc[this_cluster_idxs, cluster_id_col].unique():
                for sub_cluster_id in set(clusters.labels_):
                    sub_cluster_mask = (self.df[cluster_id_col]==sub_cluster_id) & (self.df[use_cluster_id_col]==cluster_id)
                    #top20_1grams = tfidf_df.loc[self.df[cluster_id_col].loc[sub_cluster_mask].index, 
                    top20_1grams = tfidf_df.loc[self.df.loc[sub_cluster_mask].index, 
                                                [col for col in tfidf_df.columns if '-' not in col]]\
                                            .mean().sort_values(ascending=False)[:20]
                    cluster_lbl = f'{top20_1grams.iloc[0]:.4f}-' + '-'.join(top20_1grams.index)
                    self.df.loc[sub_cluster_mask, cluster_lbl_col] = cluster_lbl
                if self.verbose:
                    print(f"Term labels complete.")

            # Drop unclustered bills from the sample
            if drop_unclustered:
                if self.verbose:
                    print(f"Dropping unclustered bills from the sample...")
                samp_size = self.df.shape[0]
                self.df = self.df.loc[self.df[cluster_id_col] != -1].copy()
                samp_size2 = self.df.shape[0]
                if self.verbose:
                    print(f"{samp_size - samp_size2} rows dropped from the sample.\n")
        
        # Format column labels
        self.df[cluster_id_col] = self.df[cluster_id_col].astype(int)


    def make_base_clusters(self,
                           force: bool = True,
                           drop_unclustered: bool = True,
                           filter_bow: bool = True,
                           stem_bow: bool = True,
                           gram_n: int = 3,
                           min_df: float = 0.005,
                           max_df: float = 0.05,
                           tfidf_norm: bool | Literal['max'] = True,
                           cluster_method: Literal['dbscan', 'hdbscan', 'optics'] = 'dbscan',
                           **cluster_params):
        """
        First-order clustering.

        Uses stemmed (truncated) versions of bill vocabularies
        in order to maximize term-matching and topic generalization.
        
        The collection of word stems in each bill text in the sample 
        is concatenated into hyphen-delimited n-grams, so common 
        sequences of up to n word-roots are included in the vector vocabulary.

        TF-IDF (Term Frequency - Inverse Document Frequency) vectors
        are created using the collection vocabulary with Document Frequency
        scores between min_df and max_df.

        The TF-IDF vectors are then normalized and compared to identify 
        clusters of bills by word stems (topic). 

        Parameters
        ----------
        gram_n : int, default 4
            Bag of Words (BoW) includes stemmed (truncated) n-grams
            for n = 1 to gram_n.

        min_df : float, default 0.0
            Minimum document frequency value for inclusion in the
            TF-IDF vocabulary vector.

        max_df : float, default 1.0
            Maximum document frequency value for inclusion in the
            TF-IDF vocabulary vector.

        min_cos : float, default 0.0
            Minimum cosine similarity for inclusion in a cluster.
        """
        if self.verbose:
            print("Begin feature processing...")
        self._make_features(level=1,
                            use_level=1,
                            force=force,
                            filter_bow=filter_bow,
                            stem_bow=stem_bow,
                            gram_n=gram_n,
                            min_df=min_df,
                            max_df=max_df,
                            tfidf_norm=tfidf_norm)
        
        if self.verbose:
            print("Feature processing complete.\n\nBegin cluster processing...")
        self._make_clusters(level=1,
                            use_level=1,
                            drop_unclustered=drop_unclustered,
                            cluster_method=cluster_method,
                            **cluster_params)
        if self.verbose:
            print("Cluster processing complete.")
        

    def make_sub_clusters(self,
                          level: int = 2,
                          use_level: int = 1,
                          use_clusters: int | list[int] | None = None,
                          use_top_n_clusters: int | None = None,
                          top_n_by: Literal['size', 'score', 'weighted'] = 'size',
                          force: bool = True,
                          drop_unclustered: bool = False,
                          filter_bow: bool = True,
                          stem_bow: bool = False,
                          gram_n: int = 5,
                          min_df: float = 0.05,
                          max_df: float = 1.0,
                          tfidf_norm: bool | Literal['max'] = False,
                          cluster_method: Literal['dbscan', 'hdbscan', 'optics'] = 'dbscan',
                          **cluster_params):
        """
        for cluster in use_clusters
            > ._make_features()
            > ._make_clusters()

        Second-order, vocabulary-level clustering. This function
        operates on each of the n most-populated topics produced
        by make_topic_clusters().

        Uses full versions of bill vocabularies in order to 
        particularize term-matching by specific terms and phrases.
        
        The collection of words in each bill text in the sample 
        is concatenated into hyphen-delimited n-grams, so common 
        sequences of up to n words are included in the vector vocabulary.

        TF-IDF (Term Frequency - Inverse Document Frequency) vectors
        are created using the collection vocabulary with Document Frequency
        scores between min_df and max_df.

        The TF-IDF vectors are then compared to identify clusters of bills 
        by their vocabularies. 


        Parameters
        ----------
        level : int, default 2
            The level to assign to the new features/clusters; must be an 
            integer greater than (but not including) 1.

        use_level : int, default 1
            The level of clustering to use as the basis for sub-clustering.

        use_clusters : int or list of int, optional
            Optional id or ids of the cluster(s) to analyze for sub-clusters;
            default includes all available clusters at the specified level (use_level=).
            NOTE: If provided, overrides use_top_n_clusters and top_n_by.

        use_top_n_clusters : int, optional
            If provided, perform subclustering on the top n clusters from
            the specified level (use_level=).
            NOTE: Parameter use_clusters overrides this one.

        top_n_by : one of: {'size','score','weighted'}, default 'size'
            Method by which to sort clusters for use_top_n_clusters.
            > 'size' = number of documents in each cluster
            > 'score' = highest average TF-IDF term score in each cluster
            > 'weighted' = (size * score) for each cluster
            NOTE: Parameter use_clusters overrides this one.

        force : bool, default True
            If True, produce document features from scratch;
            if False, reuses TF-IDF vectors from specified level (use_level=).

        gram_n : int, default 5
            Bag of Words (BoW) includes n-grams for n = 1 to gram_n.

        min_df : float, default 0.0
            Minimum document frequency value for inclusion in the
            TF-IDF vocabulary vector.

        max_df : float, default 1.0
            Maximum document frequency value for inclusion in the
            TF-IDF vocabulary vector.

        """
        if not (isinstance(level, int) and (level > 1)):
            raise ValueError(f"Invalid parameter {level=} - must be int greater than 1")
        if not (isinstance(use_level, int) and (use_level < level)):
            raise ValueError(f"Invalid parameter {use_level=} - must be int, less than {level=}")
        
        # Labels for new features
        tfidf_col = f'lvl{level}_tfidf'
        cluster_id_col = f'lvl{level}_cluster'
        cluster_lbl_col = f'lvl{level}_cluster_vocab'
        # Labels for basis features
        use_tfidf_col = f'lvl{use_level}_tfidf'
        use_cluster_id_col = f'lvl{use_level}_cluster'
        use_cluster_lbl_col = f'lvl{use_level}_cluster_vocab'

        if isinstance(use_clusters, int):
            prior_clusters = [use_clusters]
        elif isinstance(use_clusters, list):
            prior_clusters = [int(c_id) for c_id in use_clusters if str(c_id).isnumeric()]
        elif isinstance(use_top_n_clusters, int):
            sort_by = top_n_by.strip().lower() 
            if sort_by == 'size':
                prior_clusters = self.df[use_cluster_id_col]\
                                   .value_counts()[:use_top_n_clusters]\
                                   .index.tolist()
            elif sort_by == 'score':
                prior_clusters = self.df[[use_cluster_id_col,use_cluster_lbl_col]]\
                                   .drop_duplicates()\
                                   .sort_values(by=use_cluster_lbl_col, ascending=False)[use_cluster_id_col]\
                                   .values[:use_top_n_clusters]
            elif sort_by == 'weighted':
                pass # TODO(?)
            else:
                raise ValueError(f"Invalid parameter {top_n_by=} - must be one of 'size', 'score', 'weighted'")
        else:
            prior_clusters = self.df[use_cluster_id_col].unique().tolist()

        if force:
            if self.verbose:
                print("Begin feature reprocessing...")
            self._make_features(level=level,
                                use_level=use_level,
                                use_clusters=prior_clusters,
                                filter_bow=filter_bow,
                                stem_bow=stem_bow,
                                gram_n=gram_n,
                                min_df=min_df,
                                max_df=max_df,
                                tfidf_norm=tfidf_norm)
            if self.verbose:
                print("Feature reprocessing complete.\n\n")
        else:
            if self.verbose:
                print(f"Reusing features from Level {use_level}\n\n")
            
        if self.verbose:
            print("Begin cluster processing...")
        self._make_clusters(level=level,
                            use_level=use_level,
                            use_clusters=prior_clusters,
                            drop_unclustered=drop_unclustered,
                            cluster_method=cluster_method,
                            **cluster_params)
        if self.verbose:
            print("Cluster processing complete.")
            

    def get_cluster_centroids(self):
        """
        Get the average tf-idf vector defining each cluster.
        """
        pass


    def get_cluster_vocab(self,
                          level: Literal['topic', 'cluster'],
                          idx: int):
        """
        Get the defining vocabulary of each cluster.
        TODO

        Parameters
        ----------
        level : {'topic', 'cluster'}
            Clustering level to analyze.
        """
        pass


    def get_cluster_bills(self):
        """
        Get the bills in each cluster
        """
        pass


    def update_mongo(self):
        """
        Update the sample collection in MongoDB with the 
        features currently in memory.
        """
        # FIXME: Need to convert Series vectors to dict
        # FIXME: Need to convert dict back to Series on init
        for record in self.df.to_dict('records'):
            self.COLL.update_one({'_id':{'$eq':record['bill_id']}}, {'$set': record})

    