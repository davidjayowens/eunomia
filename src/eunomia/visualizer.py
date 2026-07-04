from __future__ import annotations

from typing import Literal
from collections import Counter
import re

import numpy as np
import pandas as pd

from sklearn.preprocessing import minmax_scale
from sklearn.decomposition import PCA

from diff_match_patch import diff_match_patch
import networkx as nx
from PIL import Image

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.express.colors import sample_colorscale


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
        'cluster_col',  # str
        'tfidf_col',    # str
        'max_feats',    # int
        'scale_scope',  # str
        '_polar_df',    # pandas DataFrame
        '_pca_model',   # pandas DataFrame
        'verbose',      # bool
    )

    def __init__(   self,
                    df: pd.DataFrame,
                    cluster_col: str = 'lvl1_cluster', 
                    tfidf_col: str = 'lvl1_tfidf', 
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
                        cluster_col: str | None = None,
                        tfidf_col: str | None = None,
                        max_feats: int | None = None,
                        scale_scope: str | None = None):
        """ 
        Update the object's polar_df with new parameters.
        """
        df_cols = self.df.columns.tolist()

        if cluster_col:
            if cluster_col not in df_cols:
                raise VisualizeError(f"Invalid value for parameter: {cluster_col=}\nMust be one of {df_cols}")
            
            self.cluster_col = cluster_col
        
        if tfidf_col:
            if tfidf_col not in df_cols:
                raise VisualizeError(f"Invalid value for parameter: {tfidf_col=}\nMust be one of {df_cols}")
            
            self.tfidf_col = tfidf_col

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
                            vertical_spacing=kwargs.get('vertical_spacing',0))

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
        if kwargs.get('autosize') or (not kwargs.get('width') and not kwargs.get('height')):
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

class PlotDocs:
    __slots__ = (
        '_df',                  # pandas DataFrame
        'base_cluster_col',     # str
        'sub_cluster_col',      # str
        'text_col',             # str

        'verbose'               # bool
    )

    def __init__(self,
                 df: pd.DataFrame,
                 base_cluster_col: str = 'lvl1_cluster',
                 sub_cluster_col: str | None = None,
                 text_col: str = 'text_body',

                 verbose: bool = False):
        """
        PlotDocs creates visual displays of documents, highlighting similarities
        and differences, to facilitate analysis of document clusters.

        Parameters
        ----------
        df : pandas DataFrame
            Clustered data, as produced by eunomia.clustering.DocCluster.make_doc_df().
            > Features:
            - lvl1_cluster
            - (lvl2_cluster)
            - text_body

        """
        self.verbose = verbose
        self.df = df

        # Construct graph and image features
        self.update_doc_feats(  base_cluster_col=base_cluster_col,
                                sub_cluster_col=sub_cluster_col,
                                text_col=text_col)

    # END OF __init__

    ###################################
    ##    DocDiff Special Methods    ##
    ###################################

    def __repr__(self):
        return(f"DocDiff(TBD)")
    
    def __str__(self) -> str:
        return(f"DocDiff\n==========\n"
               f"TBD"
               )


    ###############################
    ##    DocDiff Attributes:    ##
    ##      Getters & Setters    ##
    ###############################

    @property
    def df(self) -> pd.DataFrame:
        return(self._df)
    @df.setter
    def df(self, new_data):
        if not isinstance(new_data, pd.DataFrame):
            raise VisualizeError(f"Invalid object of type {type(new_data)}\nMust be pandas DataFrame.")
        
        # Enforce RangeIndex
        self._df = new_data.reset_index(drop=True).copy()


    ################################
    ##    DocDiff Attributes:     ##
    ##       Update Methods       ##
    ################################

    @staticmethod
    def _filter_doc(txt:str) -> str:
        """ Apply simple text cleaning to improve document matching. """
        filters = {
            r'\d+': ' ',                        # all numbers
            r'(?<=\s)[^\w\s]+(?=\s)': ' ',      # standalone punctuation
            r'(?<=\s)\w(?=\s)': ' '             # single characters
            }

        # Apply filters
        for patt,repl in filters.items():
            txt = re.sub(patt, repl, txt)
        # Trim excess whitespace
        txt = re.sub(r'\s+', ' ', txt).strip()      

        return(txt)
    

    @staticmethod
    def _get_diff(txt1:str, txt2:str):
        """ Get diff_match_patch semantic diff of two texts. """
        # Initialize dmp
        dmp = diff_match_patch()
        
        # Create doc comparison
        diff = dmp.diff_main(txt1, txt2)
        dmp.diff_cleanupSemantic(diff)

        return(diff)


    @staticmethod
    def _get_match(diff) -> str:
        """ Join diff results back into a contiguous string """
        matchstr = ' '.join([d[1] for d in diff if d[0]==0])
        matchstr = re.sub(r'\s+', ' ', matchstr).strip()
        
        return(matchstr)


    @staticmethod
    def _get_match_lens(diff, nomatch_mean=True) -> list[tuple]:
        """ 
        Return (approximate) lengths of the matching vs non-matching text segments. 
        
        Returns list of tuples like: [('r'/'g', len)]
            - 'r' for red (nonmatching)
            - 'g' for green (matching)
        
        nomatch_mean : bool, default True
            If True, when deleted text from the left doc is accompanied by
            added/inserted text from the right doc, average their lengths to determine
            the length of the non-matching segment.

            If False, only the length of the addition in the right doc is used.

            NOTE: Since the intersection of multiple texts results in fewer
            deletions from the current intersection and mostly additions in the
            newly compared doc, continuous averaging can make the final intersection
            appear to be more of a match than it truly is. 
            
            Use True here if only comparing two documents, but use False when 
            doing a series of cumulative intersections with n>2 docs. Important 
            to remember that the resulting intersection of each comparison 
            should be the "left" doc in each subsequent comparison made.

        """
        skip = False
        lens = []
        max_idx = len(diff) - 1
        for i,t in enumerate(diff):
            if t[0] == 0:   # Matching string
                lens.append(('g', len(t[1])))
                skip = False
            elif t[0] == -1: # Deletion from string 1
                if (i < max_idx) and (diff[i+1][0] == 1):  # Followed by addition to string 2
                    if nomatch_mean:
                        lens.append(('r', int(np.mean([len(t[1]), len(diff[i+1][1])]))))
                    else:
                        lens.append(('r', len(diff[i+1][1])))
                    skip = True
                else:   # Not followed by addition to string 2
                    lens.append(('r', len(t[1])))
                    skip = False
            elif not skip:   # t[0]==1 -> Addition to string 2
                lens.append(('r', len(t[1])))
                skip = False
        
        return(lens)


    @staticmethod
    def _make_img(match_lens) -> Image:
        """ Use results of match_lens() to produce a r/g/k image representation. """
        colors = {
            'k': (0,0,0),   # black
            'r': (255,0,0), # red
            'g': (0,255,0)  # green
        }

        # Find size of each block
        total_len = sum([t[1] for t in match_lens])
        size = total_len / 300

        # Flat list of char colors
        charlist = []
        for tup in match_lens:
            charlist.extend([tup[0]] * tup[1])

        # 300 segments (5*5px) = 15 wide, 20 tall
        pixels = []
        i = 1
        maxblock = 0
        while maxblock < 300:
            if size < 1.0:  # Each char -> multiple blocks
                blocksize = 1.0/size
                minblock = int((i-1)*blocksize)+1
                maxblock = int((i)*blocksize)
                top_char = charlist[i-1]
                pixels.extend([top_char]*((maxblock-minblock)+1))
            else:           # Each block -> multiple chars
                maxblock = i
                start = int((i-1)*size)
                stop = int(i*size) + (1 if maxblock==300 else 0)
                top_char = Counter(charlist[start:stop]).most_common()[0][0]
                pixels.append(top_char)
            i += 1
        # pixels -> 300 'r'/'g' chars

        # Add black borders: 17 segments top & bottom, 1 segment left & right
        segdim = 5  # height & width
        border_px = []
        # Top border
        border_px.extend(['k']*segdim*17*segdim)  # 5 px wide segment * 17 segments * 5 px high
        
        # Text rows
        textrowsegs = 15
        for i in range(20):
            this_row = []
            # Left border
            this_row.extend(['k']*segdim)
            # Text pixels
            for px in pixels[(i*textrowsegs):(i*textrowsegs)+textrowsegs]:
                this_row.extend([px]*segdim)
            # Right border
            this_row.extend(['k']*segdim)
            border_px.extend(this_row*segdim)

        # Bottom border:
        border_px.extend(['k']*segdim*17*segdim)

        # Replace r/g/k with actual RGB pixel values, reshape results
        actual_px = [colors[px] for px in border_px]
        px_arr = np.array(actual_px, dtype=np.uint8).reshape(110,85,3)

        # Turn pixels into image
        img = Image.fromarray(px_arr, mode='RGB')

        return(img)


    def update_doc_feats(self,
                         base_cluster_col: str | None = None,
                         sub_cluster_col: str | None = None,
                         text_col: str | None = None):
        """
        Update the object's doc-related features with new parameters.
        """
        df_cols = self.df.columns.tolist()

        if base_cluster_col:
            if base_cluster_col not in df_cols:
                raise VisualizeError(f"Invalid value for parameter: {base_cluster_col=}\nMust be one of {df_cols}")
            
            self.base_cluster_col = base_cluster_col

        if sub_cluster_col:
            if sub_cluster_col not in df_cols:
                raise VisualizeError(f"Invalid value for parameter: {sub_cluster_col=}\nMust be one of {df_cols}")
            
            self.sub_cluster_col = sub_cluster_col

        if text_col:
            if text_col not in df_cols:
                raise VisualizeError(f"Invalid value for parameter: {text_col=}\nMust be one of {df_cols}")
            
            self.text_col = text_col





    ###################################
    ##    DocDiff Primary Methods    ##
    ###################################

    def diff_img(self, *txts:str):
        """ 
        Wrapper for image pipeline:
        - _filter_doc() on individual texts
        - _get_diff() cumulatively compares all texts in cluster
        - _get_match() gets intersection of all texts in cluster
        - _get_match_lens() determines lengths of matching vs non-matching segments
        - _make_img() produces RGK image representation of doc intersection
        """
        if not len(txts)>=2:
            raise ValueError("Must provide at least 2 strings to compare.")
        
        # Basic text cleaning
        t1 = filter_doc(txts[0])
        t2 = filter_doc(txts[1])

        diff = get_diff(t1, t2)
        m_str = get_match(diff)

        # Continuously take the intersection of all texts
        for txt in txts[2:]:
            _t = filter_doc(txt)
            diff = get_diff(m_str, _t)
            m_str = get_match(diff)
        
        # Get lengths of matching vs nonmatching sections
        m_lens = match_lens(diff, nomatch_mean=False)

        # Generate final image representation
        img = img_gen(m_lens)

        return(img)


    def make_nx_graph(  self,
                        cluster_id: str,
                        images: dict,
                        subcluster_ids: list | None = None,
                        text_counts: dict | None = None,
                        scale: int | None = None):
        """
        images : dict
            Like {'cluster'/'subcluster': Image}
        """
        # Cluster graph
        G = nx.Graph()
        
        # Add nodes
        nodes = subcluster_ids or [cluster_id]
        G.add_nodes_from(nodes)
        
        # Create edges - nodes are fully connected
        G = nx.complete_graph(G)

        # Assign attributes to graph & nodes
        if not scale:
            scale = 0.85 * len(nodes)

        g_attr = {  
            'cluster_id': cluster_id,
            'scale': scale
            }
        for k,v in g_attr.items():
            G.graph[k] = v

        node_attrs = {
            n: {
                'label': f"<b>Cluster:</b> {cluster_id}"
                        f"{f'<br><b>Subcluster:</b> {n}' if subcluster_ids else ''}"
                        f"{f'<br><b># Texts: {text_counts[n]}</b>' if text_counts else ''}",
                'img': images[n]
                } 
            for n in nodes
            }
        nx.set_node_attributes(G, node_attrs)

        return(G)

    """
    subgraphs = [
    make_nx_graph(
        cluster_id=id1,
        images={id2: diff_img(*doc2_df.loc[(doc2_df.lvl1_cluster == id1)
                                         & (doc2_df.lvl2_cluster == id2), 'text_body'].tolist())
                for id2 in doc2_df.loc[(doc2_df.lvl1_cluster == id1), 'lvl2_cluster'].unique()
                },
        subcluster_ids = [id2 for id2 in doc2_df.loc[(doc2_df.lvl1_cluster == id1), 'lvl2_cluster'].unique()],
        text_counts = {id2: doc2_df.loc[(doc2_df.lvl1_cluster == id1)
                                      & (doc2_df.lvl2_cluster == id2)].shape[0]
                for id2 in doc2_df.loc[(doc2_df.lvl1_cluster == id1), 'lvl2_cluster'].unique()                         
                },
        #scale = 0.5,
        #bubble_size = 300
        )
        for id1 in doc2_df['lvl1_cluster'].unique()
    ]
    """

    def plot(   self,
                graphs: list,
                max_subplot_cols: int = 3,
                plot_range: int = 1,
                img_scale: int | None = None,
                marker_scale: int | None = None,
                bubble_scale: int = 1000,
                inplace:bool=True,
                **kwargs):
        """ Plot the clusters """
        
        num_clusters = len(graphs)

        cols = min(max_subplot_cols, num_clusters)
        rows = int(num_clusters / max_subplot_cols) + (num_clusters%max_subplot_cols > 0)
        remainder = num_clusters % max_subplot_cols

        fig = make_subplots(rows=rows, cols=cols,
                            vertical_spacing=0, horizontal_spacing=0)
        graph_i = 0
        for r in range(1, rows+1):
            # TODO: This may be over-engineered; may need to simply make the 
            # plotting conditional on if graph_i < num_clusters, but still style axes
            # for subplots without actual graphs
            #for c in range(1, cols+1) if not remainder \
            #    else range(1, cols+1) if (r<rows) \
            #    else range(1, remainder+1):
            for c in range(1, cols+1):
                if graph_i < num_clusters:
                    this_graph = graphs[graph_i]
                    # Establish node coords
                    _ = nx.planar_layout(this_graph, 
                                        center = (0, 0), 
                                        scale = this_graph.graph['scale'],
                                        store_pos_as = 'pos')

                    # Plot edges between subclusters
                    edge_x = []
                    edge_y = []
                    for edge in this_graph.edges():
                        x0, y0 = this_graph.nodes[edge[0]]['pos']
                        x1, y1 = this_graph.nodes[edge[1]]['pos']
                        edge_x.append(x0)
                        edge_x.append(x1)
                        edge_x.append(None)
                        edge_y.append(y0)
                        edge_y.append(y1)
                        edge_y.append(None)
                    # Add edges to subplot
                    fig.add_trace(go.Scatter(
                            x=edge_x, y=edge_y,
                            line=dict(width=0.5, color='black'),
                            hoverinfo='none',
                            mode='lines',
                            zorder=1
                        ), row=r, col=c)
                    
                    # Plot images w/ underlying annotated markers
                    node_x = []
                    node_y = []
                    labels = []
                    for node in this_graph.nodes():
                        x,y = this_graph.nodes[node]['pos']
                        node_x.append(x)
                        node_y.append(y)
                        labels.append(this_graph.nodes[node]['label'])
                        # Add image to subplot
                        fig.add_layout_image(
                                dict(x=x,                y=y,
                                    xref='x',           yref='y',
                                    xanchor='center',   yanchor='middle',
                                    sizing='contain',
                                    opacity=1.0,
                                    layer='above',
                                    source=this_graph.nodes[node]['img']
                            ), row=r, col=c)
                    # Add annotated markers to subplot
                    fig.add_trace(go.Scatter(
                        x=node_x, y=node_y,
                        mode='markers',
                        marker=dict(
                            showscale=False,
                            color='black',
                            size=marker_scale,
                            sizemode='diameter'
                        ),
                        hoverinfo='text',
                        text=labels,
                        zorder=2,
                    ), row=r, col=c)
                # END OF graph-specific plotting steps
                
                # Add circle around cluster/to fill subplot window
                fig.add_trace(go.Scatter(
                    x = [0],   
                    y = [0],
                    mode='markers',
                    marker=dict(
                        size=bubble_scale,
                        sizemode='diameter',
                    ),
                    hoverinfo='none',
                    zorder=0
                ), row=r, col=c)

                # Set axes
                fig.update_xaxes(range=[-plot_range*1.5,plot_range*1.5], 
                                visible=False, 
                                row=r, col=c)
                fig.update_yaxes(range=[-plot_range,plot_range], 
                                #scaleanchor = "x", 
                                #scaleratio = 1,
                                visible=False, 
                                row=r, col=c)
                graph_i += 1

        fig.update_layout(template='plotly', 
                            showlegend=False,
                            #width=900, height=600,
                            #margin={'l':10,'r':10,'t':10,'b':10},
                            hovermode='closest')
        fig.update_layout_images(sizex=img_scale, sizey=img_scale)
        
        # Optional settings
        if kwargs.get('autosize') or (not kwargs.get('width') and not kwargs.get('height')):
            fig.update_layout(autosize=True)
        else:
            fig.update_layout(autosize=False)
            if kwargs.get('width'):
                fig.update_layout(width=kwargs.get('width'))
            if kwargs.get('height'):
                fig.update_layout(height=kwargs.get('height'))

        fig.update_layout(margin=kwargs.get('margin', {'l':10,'r':10,'t':10,'b':10}))

        if inplace:
            fig.show(config={'doubleClick': 'reset', 'displayModeBar': False})
        else:
            return(fig)
        
# END OF DocDiff class