from __future__ import annotations

from typing import Literal
import numpy as np
import pandas as pd

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.express.colors import sample_colorscale

from sklearn.preprocessing import minmax_scale
from sklearn.decomposition import PCA

import difflib
import re
#from nltk.tokenize import sent_tokenize
import networkx as nx

import logging
log = logging.getLogger(__name__)

def _log(msg:str, verbose:bool=False) -> None:
    """ Log msg at debug level and optionally print to stdout. """
    log.debug(msg, stacklevel=2)
    if verbose:
        print(msg)

class VisualizeError(Exception):
    def __init__(self, msg):
        log.error("Eunomia | VisualizeError exception encountered:\n" + msg, stacklevel=2)
        super().__init__(msg)


##############################
##    Visualize Clusters    ##
##############################

class PlotPolars:
    __slots__ = (
        '_df',          # pandas DataFrame
        'tfidf_col',    # str
        'cluster_col',  # str
        'max_feats',    # int
        'scale_scope',  # str
        '_polar_df',    # pandas DataFrame
        '_pca_model',   # pandas DataFrame
        'verbose',      # bool
    )

    def __init__(   self,
                    df: pd.DataFrame,
                    tfidf_col: str, 
                    cluster_col: str, 
                    max_feats: int = 16,
                    scale_scope: Literal['collection', 'cluster', 'feature', 'vector'] = 'collection',
                    verbose: bool = False):
        """
        PlotPolars creates radial plots to visualize document clusters. It
        simplifies the Plotly interface, while still giving users the flexibility
        to customize plots with fine-tuned adjustments.

        Parameters
        ----------
        df : pandas DataFrame
            Clustered data, as produced by eunomia.clustering.DocCluster.

        tfidf_col : str
            Label of the tf-idf column used in clustering.

        cluster_col : str
            Label of the column containing cluster IDs.

        max_feats : int, default 16
            Reduce the number of tf-idf features to max_feats (via PCA)
            for visualization. Actual number of features will be 
            min(max_feats, n_samples, n_features).

        scale_scope : one of {'collection', 'cluster', 'feature', 'vector'}, default 'collection'
            Before visualizing the feature-reduced tf-idf vectors, data is 
            adjusted to a [0,1] scale in preparation for plotting.

            The scale_scope param defines how this scaling is applied to the results,
            such that 0 and 1 correspond to:
                > 'collection' - the absolute min/max values across all vectors and all clusters
                > 'cluster' - the min/max values on a per-cluster basis (across all vectors & features)
                > 'feature' - the min/max values on a per-feature basis (across all vectors & clusters)
                > 'vector' - the min/max values on a per-vector basis (across all features)

            NOTE: Scales other than 'collection' can produce more aesthetically interesting
            plots, but some comparative value is necessarily lost. 
            
        verbose: bool, default False
            If True, prints updates to stdout during processing.

        
        Examples of data scaling
        ------------------------

                Sample data
                -----------
                cluster 1:
                    vector 1: [0.01,        0.02,           0.05]
                    vector 2: [0.15,        0.30,           0.75]
                cluster 2:
                    vector 1: [0.50,        0.01,           0.75]
                    vector 2: [0.60,        0.15,           0.98]

                'collection' scaling
                --------------------
                cluster 1:
                    vector 1: [0.0,         0.01030928,     0.04123711]
                    vector 2: [0.1443299,   0.29896907,     0.7628866]
                cluster 2:
                    vector 1: [0.50515464,  0.0,            0.7628866]
                    vector 2: [0.60824742,  0.1443299,      1.0]
                Pros:
                > Minimally transformative of the underlying data
                > Relative values are still proportional to each other, 
                  across all vectors/features/clusters
                Cons:
                > Some flattening of data, less dynamic range on a
                  per-vector/feature/cluster basis
                
                'cluster' scaling
                -----------------
                cluster 1:
                    vector 1: [0.0,         0.01351351, 0.05405405]
                    vector 2: [0.18918919,  0.39189189, 1.0]
                cluster 2:
                    vector 1: [0.50515464,  0.0,        0.7628866]
                    vector 2: [0.60824742,  0.1443299,  1.0]
                Pros:
                > Preserves relationships of values across vectors and 
                  features within a cluster
                > Greater dynamic range within a cluster; values not flattened 
                  by extremes in other clusters
                Cons:
                > Reduced ability to compare values across clusters

                'feature' scaling
                -----------------
                cluster 1:
                    vector 1: [0.0,         0.03448276,     0.0]
                    vector 2: [0.23728814,  1.0,            0.75268817]
                cluster 2:
                    vector 1: [0.83050847,  0.0,            0.75268817]
                    vector 2: [1.0,         0.48275862,     1.0]
                Pros:
                > Within each feature, shows which vectors & clusters have 
                  the highest/lowest values
                Cons:
                > Loses comparative value between features across vectors/clusters

                'vector' scaling
                ----------------
                cluster 1:
                    vector 1: [0.0,         0.25,   1.0]
                    vector 2: [0.0,         0.25,   1.0]
                cluster 2:
                    vector 1: [0.66216216,  0.0,    1.0]
                    vector 2: [0.54216867,  0.0,    1.0]
                Pros:
                > Within each vector, shows which features have the 
                  highest/lowest values
                Cons:
                > Loses comparative value between vectors/clusters

        """
        self.verbose = verbose
        self.df = df

        # Construct DataFrames used in plotting and labeling clusters
        self.update_polar_df(   tfidf_col=tfidf_col,
                                cluster_col=cluster_col,
                                max_feats=max_feats,
                                scale_scope=scale_scope)
        
    # END OF __init__

    ######################################
    ##    PlotPolars Special Methods    ##
    ######################################

    def __repr__(self):
        return(f"PlotPolars(df=<pd.DataFrame>, tfidf_col={self.tfidf_col}, "
               f"cluster_col={self.cluster_col}, max_feats={self.max_feats}), "
               f"scale_scope={self.scale_scope}, verbose={self.verbose}")
    
    def __str__(self) -> str:
        return(f"PlotPolars\n==========\n"
               f"df = <pd.DataFrame>\n"
               f"tfidf_col = {self.tfidf_col}\n"
               f"cluster_col = {self.cluster_col}\n"
               f"max_feats = {self.max_feats}\n"
               f"scale_scope = {self.scale_scope}\n"
               f"verbose = {self.verbose}"
               )


    ##################################
    ##    PlotPolars Attributes:    ##
    ##      Getters & Setters       ##
    ##################################

    @property
    def df(self) -> pd.DataFrame:
        return(self._df)
    @df.setter
    def df(self, new_data):
        if not isinstance(new_data, pd.DataFrame):
            raise VisualizeError(f"Invalid object of type {type(new_data)}\nMust be pandas DataFrame.")
        
        # Enforce RangeIndex - this will be used to merge with PCA-reduced features in .plot()
        self._df = new_data.reset_index(drop=True).copy()


    ##################################
    ##    PlotPolars Attributes:    ##
    ##       Updater Methods        ##
    ##################################

    def update_polar_df(self,
                        tfidf_col: str | None = None,
                        cluster_col: str | None = None,
                        max_feats: int | None = None,
                        scale_scope: str | None = None):
        """ 
        Update the object's polar_df with new parameters.
        """
        df_cols = self.df.columns.tolist()

        if tfidf_col:
            if tfidf_col not in df_cols:
                raise VisualizeError(f"Invalid value for parameter: {tfidf_col=}\nMust be one of {df_cols}")
            
            self.tfidf_col = tfidf_col

        if cluster_col:
            if cluster_col not in df_cols:
                raise VisualizeError(f"Invalid value for parameter: {cluster_col=}\nMust be one of {df_cols}")
            
            self.cluster_col = cluster_col

        if max_feats:
            self.max_feats = int(max_feats)

        if scale_scope:
            valid_scale_scopes = ['collection', 'cluster', 'feature', 'vector']
            if scale_scope.lower() not in valid_scale_scopes:
                raise VisualizeError(f"Invalid value for parameter: {scale_scope=}\nMust be one of {valid_scale_scopes}")
            
            self.scale_scope = scale_scope.lower()

        self._make_polar_df()
        

    def _make_polar_df(self):
        """
        Transforms source data into a version ready for plotting and labeling.
        """
        # Matrix of tf-idf scores; rows = documents, columns = terms
        tfidf_df =  pd.DataFrame.from_dict(dict(zip(self.df[self.tfidf_col].index, self.df[self.tfidf_col].values))).T.fillna(0)
        tfidf_feats = tfidf_df.shape[1]

        # Reduce (by PCA) to a number of features that can be visualized
        num_feats = min(self.max_feats, tfidf_df.shape[0], tfidf_df.shape[1])
        self._pca_model = PCA(n_components=num_feats).fit(tfidf_df)
        
        pca_df = pd.DataFrame(self._pca_model.transform(tfidf_df), index=self.df.index)
        
        # Scale all data to [0,1] range
        if self.scale_scope == 'collection':
            # Scaling based on min and max values anywhere in pca_df
            min_val = pca_df.min(axis=None)
            max_val = pca_df.max(axis=None)
            diff_val = max_val - min_val

            pca_scaled = (pca_df - min_val) / diff_val

        elif self.scale_scope == 'cluster':
            pca_scaled = pca_df.head(0)

            for cluster_id in self.df[self.cluster_col].unique():
                cluster = pca_df.loc[self.df.loc[self.df[self.cluster_col]==cluster_id].index]
                                     
                min_val = cluster.min(axis=None)
                max_val = cluster.max(axis=None)
                diff_val = max_val - min_val

                cluster_scaled = (cluster - min_val) / diff_val 
                pca_scaled = pd.concat([pca_scaled, cluster_scaled])    
        
        elif self.scale_scope == 'feature':
            # Scale each feature individually (column-wise)
            min_vals = pca_df.min()
            max_vals = pca_df.max()
            diff_vals = max_vals - min_vals

            pca_scaled = pca_df.sub(min_vals, axis=1).div(diff_vals, axis=1)

        elif self.scale_scope == 'vector':
            # Scale each vector individually (row-wise)
            min_vals = pca_df.min(axis=1)
            max_vals = pca_df.max(axis=1)
            diff_vals = max_vals - min_vals

            pca_scaled = pca_df.sub(min_vals, axis=0).div(diff_vals, axis=0)

        # Assemble the version of the DF for plotting with polar coordinates
        polar_df = pd.DataFrame(pca_scaled.stack(level=0)).reset_index(level=1, names=['','pca_feat'])
        polar_df = polar_df.merge(self.df[self.cluster_col], how='left', left_index=True, right_index=True)
        polar_df.rename(columns={0:'pca_score', self.cluster_col:'cluster_id'}, inplace=True)
        for str_feat in ['pca_feat', 'cluster_id']:
            polar_df[str_feat] = polar_df[str_feat].astype(str)

        self._polar_df = polar_df


    ######################################
    ##    PlotPolars Primary Methods    ##
    ######################################
    
    def plot(   self, 
                cluster_ids: list[int|str] | None = None, 
                subplots: bool = False, 
                max_subplot_cols: int = 3, 
                labels_like: str = 'Cluster {id}',
                inplace: bool = True, 
                **kwargs):
        """

        Parameters
        ----------
        cluster_ids : list of str/int, optional
            List of cluster_id values to include in the plot.

        subplots : bool, default False
            If True, creates individual plots for each cluster;
            if False, all clusters are plotted on a single graph.

        max_subplot_cols: int, default 3
            If subplots=True, this value sets the maximum number of subplot
            columns.

        labels_like : str, default 'Cluster {id}'
            Sets how each cluster is identified on the legend. Use '{id}' 
            to include the cluster_id value.

        inplace : bool, default True
            If True, immediately displays the visualization;
            if False, returns the final plot as an object.

        **kwargs : optional settings
            > autosize : bool, automatically configure plot size, default True
                >> NOTE: autosize and width/height are mutually exclusive!
            > width : int, sets plot width, default is autosize=True
            > height : int, sets plot height, default is autosize=True
            > horizontal_spacing : float, horizontal distance between subplots if max_subplot_cols>1, default 0.0
                >> NOTE: Max value is 1 / (ncols - 1)
            > vertical_spacing : float, vertical distance between subplots if nrows>1, default 0.0
                >> NOTE: Max value is 1 / (nrows - 1)
            > margin : dict, sets plot margins
                >> EG: margin={'l':10,'r':10,'t':10,'b':10}
            > minreducedwidth : int, sets minimum (sub)plot width
            > minreducedheight : int, sets minimum (sub)plot height
            
            > template : str, plotly color template, default 'plotly_dark'
            > paper_bgcolor : str, plotly color value, fill color outside the polar plot, default black 
            > plot_bgcolor : str, plotly color value, fill color inside the polar plot, default black
            > grid_color : plotly color value, line color of the boundary/axes of the polar plot, default grey
            > colorscale : str, plotly colorscale for vector plots, default 'turbo_r'
            > colorscale_low : float, low end of the color spectrum, default 0.0
            > colorscale_high : float, high end of the color spectrum, default 1.0
            > fill_traces : bool, True fills the interior of each plot, default False
            
            > title : str, sets plot title, default "Plot of TF-IDF vectors"

            > showlegend : bool, controls the legend display, default True
            > ticklabel_nterms : int, display top n terms contributing to each PCA-reduced feature, default 0
            > ticklabel_radius : int, distance of tick labels from plot perimeter, default 1
            
            > rounded_plot : bool, False makes plat polygonal, default True
            
    
        """

        if cluster_ids is None:
            cluster_ids = self._polar_df.cluster_id.unique().tolist()
        elif not isinstance(cluster_ids, list):
            cluster_ids = [cluster_ids]

        # Cluster IDs can be provided as int or str, but are stored as str values
        cluster_idnums = [int(id) for id in cluster_ids]    # Validate that values are numeric
        sorted_cluster_ids = [str(id) for id in sorted(cluster_idnums)] # Store as strings
        num_clusters = len(sorted_cluster_ids)

        # Create color map
        discrete_colors = dict(zip( sorted_cluster_ids, 
                                    sample_colorscale(  kwargs.get('colorscale','turbo_r'), 
                                                        minmax_scale(sorted_cluster_ids), 
                                                        low=kwargs.get('colorscale_low',0.0), 
                                                        high=kwargs.get('colorscale_high',1.0)
                                                    )
                                    )
                                )
        
        if subplots:
            cols = min(max_subplot_cols, num_clusters)
            rows = int(num_clusters / max_subplot_cols) + (num_clusters%max_subplot_cols > 0)
            if (rows > 1):
                specs = [[{'type':'polar'} for _ in range(max_subplot_cols)] for _ in range(rows-1)]
                remainder = num_clusters % max_subplot_cols
                if remainder:
                    specs.append([{'type':'polar'} for _ in range(remainder)] + [{} for _ in range(max_subplot_cols - remainder)])
                else:
                    specs.append([{'type':'polar'} for _ in range(max_subplot_cols)])
            else:
                specs = [[{'type':'polar'} for _ in range(num_clusters)]]
        else:
            cols = 1
            rows = 1
            specs = [[{'type':'polar'}]]
        
        fig = make_subplots(rows=rows, cols=cols,
                            specs=specs, 
                            horizontal_spacing=kwargs.get('horizontal_spacing',0),
                            vertical_spacing=kwargs.get('vertical_spacing',0.0))

        # Add trace for each cluster
        row = 1
        col = 1
        for cluster in sorted_cluster_ids:
            cluster_df = self._polar_df.loc[self._polar_df.cluster_id==cluster]
            fig.add_trace(  go.Scatterpolar(r=cluster_df['pca_score'], 
                                            theta=cluster_df['pca_feat'],
                                            mode='lines', 
                                            name=labels_like.format(id=cluster),
                                            line_color=discrete_colors[cluster],
                                            fillcolor=discrete_colors[cluster]),
                            row=row, 
                            col=col)
            if subplots:
                col += 1
                if col > max_subplot_cols:
                    col = 1
                    row += 1

        # Style polar plots
        fig.update_polars({ 'radialaxis':{  'visible': False, 
                                            'range': [0,1]},
                            'angularaxis':{ 'showticklabels': False,
                                            'ticks': '',
                                            'rotation': 90,
                                            'direction':'clockwise'}})
        if kwargs.get('plot_bgcolor'):
            fig.update_polars({'bgcolor':kwargs.get('plot_bgcolor')})
        if kwargs.get('rounded_plot') is False:
            fig.update_polars({'gridshape': 'linear'})
            
        if kwargs.get('grid_color'):
            fig.update_polars({'angularaxis': {'gridcolor': kwargs.get('grid_color')}})
        
        # Style traces
        fig.update_traces({'hoverinfo': 'none'})
        if kwargs.get('fill_traces'):
            fig.update_traces({'fill': 'toself'})

        # Style layout
        fig.update_layout(  title = kwargs.get('title', 'Plot of TF-IDF vectors'),
                            showlegend = kwargs.get('showlegend',True),
                            template = kwargs.get('template','plotly_dark'),
                        )
        if kwargs.get('autosize') and not kwargs.get('width') and not kwargs.get('height'):
            fig.update_layout(autosize=True)
        else:
            fig.update_layout(autosize=False)
            if kwargs.get('width'):
                fig.update_layout(width=kwargs.get('width'))
            if kwargs.get('height'):
                fig.update_layout(height=kwargs.get('height'))
        if kwargs.get('margin'):
            # like: margin={'l':10,'r':10,'t':10,'b':10}
            fig.update_layout(margin=kwargs.get('margin'))
        if kwargs.get('minreducedwidth'):
            fig.update_layout(minreducedwidth=kwargs.get('minreducedwidth'))
        if kwargs.get('minreducedheight'):
            fig.update_layout(minreducedheight=kwargs.get('minreducedheight')) 
        if kwargs.get('paper_bgcolor'):
            fig.update_layout(paper_bgcolor=kwargs.get('paper_bgcolor'))

        if kwargs.get('ticklabel_nterms'):
            try:
                n_terms = int(kwargs.get('ticklabel_nterms'))                
                if n_terms < 0:
                    raise Exception
                
                label_dist = int(kwargs.get('ticklabel_radius', 0))
                if label_dist < 0:
                    raise Exception

            except:
                raise VisualizeError(f'Invalid parameters:\n'
                                     f'ticklabel_nterms = {kwargs.get('ticklabel_nterms')}\n'
                                     f'ticklabel_radius = {kwargs.get('ticklabel_radius')}\n'
                                      'Values must be non-negative integers (>= 0).')
            if n_terms > 0:
                tick_labels = self._pca_feature_labels(n_terms=n_terms)

                fig.update_polars({'angularaxis': {
                                        'showticklabels': True,
                                        'ticks': 'outside',
                                        'labelalias': tick_labels,
                                        'ticklen': label_dist
                                    }})

        if inplace:
            fig.show()
        else:
            return(fig)


    def _pca_feature_labels(self, 
                            n_terms: int = 3, 
                            abs: bool = True):
        """ 
        Get the n_terms number of feature labels from the source data
        that contribute the most variability to each feature in the 
        PCA feature-reduced plot data.

        Paramters
        ---------
        n_terms : int, default 3
            The number of terms/labels to return for each output feature.

        abs : bool, default True
            If True, takes the absolute value of the variability scores when
            determining the top contributors; if False, only positive contribution
            is considered.

        """
        # Feature names are uniform across all the vectors; grabbing the first one arbitrarily
        feat_names = self.df.at[0,self.tfidf_col].index
        
        components = self._pca_model.components_.copy()
        if abs:
            components = np.abs(components)
        
        n_components = len(components)

        labels = []
        for i in range(n_components):
            labels.append(', '.join(feat_names[components[i].argsort()[-1:-(n_terms+1):-1]]))
        
        results = {i:lbl for i,lbl in zip(range(n_components), labels)}

        return(results)

# END OF PlotPolars class




###############################################
##    Highlight Document Text Differences    ##
###############################################

class DocDiff:
    def __init__(self,
                 df: pd.DataFrame,
                 ):
        """
        DocDiff creates visual displays of documents, highlighting similarities
        and differences, to facilitate analysis of document clusters.

        Parameters
        ----------
        df : pandas DataFrame
            Clustered data, as produced by eunomia.clustering.DocCluster.

        """
        self._df = df.copy()



    def filter_nums(text):
        # Filter numbers and non-terminal punctuation
        pattern1 = re.compile(r'(\d+)') # numbers
        pattern2 = re.compile(r'["#$%&\'()*+,\-/\\:;<=>@[\]^_`{|}~]') # all punctuation except for: . ! ?
        pattern3 = re.compile(r'\b(?=[mdclxvi])m*(c[md]|d?c{0,3})(x[cl]|l?x{0,3})(i[xv]|v?i{0,3})\b') # roman numerals
        pattern4 = re.compile(r'\ +')

        text_filtered = text
        for pattern in [pattern1, pattern2, pattern3, pattern4]:
            text_filtered = re.sub(pattern, ' ', text_filtered)
        
        return(text_filtered)


    def show_diff(lines1, lines2):
        #lines1 = string1.splitlines()
        #lines2 = string2.splitlines()

        differ = difflib.Differ()
        diff = differ.compare(lines1, lines2)

        for line in diff:
            if line.startswith("- "):
                print(f"\033[31m{line}\033[0m")  # Red for removals
            elif line.startswith("+ "):
                print(f"\033[32m{line}\033[0m")  # Green for additions
            elif line.startswith("? "):
                print(f"\033[33m{line}\033[0m")  # Yellow for hints
            else:
                print(line)


    def show_unified_diff(lines1, lines2):
        #lines1 = string1.splitlines()
        #lines2 = string2.splitlines()

        diff = difflib.unified_diff(lines1, lines2, lineterm="")
        for line in diff:
            if line.startswith("-"):
                print(f"\033[31m{line}\033[0m")  # Red for removals
            elif line.startswith("+"):
                print(f"\033[32m{line}\033[0m")  # Green for additions
            else:
                print(line)


    def doc_diff(df, idx1, idx2, text_col):
        text1 = sent_tokenize(filter_nums(df.loc[idx1, text_col]))
        text2 = sent_tokenize(filter_nums(df.loc[idx2, text_col]))

        return(show_diff(text1, text2))