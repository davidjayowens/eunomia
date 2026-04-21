import pandas as pd

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.express.colors import sample_colorscale

from sklearn.preprocessing import minmax_scale
from sklearn.decomposition import PCA

import difflib
import re
from nltk.tokenize import sent_tokenize


##############################
##    Visualize Clusters    ##
##############################

class PlotPolars:
    def __init__(   self,
                    df: pd.DataFrame,
                    tfidf_col: str, 
                    cluster_col: str, 
                    num_feats: int = 16):
        """
        Visualizer for document clusters

        Parameters
        ----------
        df : pandas DataFrame
            Clustered data, as produced by eunomia.clustering.DocCluster.

        tfidf_col: str
            Label of the tf-idf column used in clustering.

        cluster_col: str
            Label of the column containing cluster IDs.

        num_feats: int, default 16
            Reduce the number of tf=idf features to num_feats (via PCA)
            for visualization.

        """
        self.df = df

        # DataFrame used in plotting and labeling clusters
        self.update_polar_df(   tfidf_col=tfidf_col,
                                cluster_col=cluster_col,
                                num_feats=num_feats)
        # Also sets self.tfidf_col, self.cluster_col, and self.num_feats
        

    def __repr__(self):
        return(f"PlotPolars(df=<pd.DataFrame>, tfidf_col={self.tfidf_col}, cluster_col={self.cluster_col}, num_feats={self.num_feats})")
    

    @property
    def df(self):
        return(self._df)
    @df.setter
    def df(self, new_df):
        if not isinstance(new_df, pd.DataFrame):
            raise ValueError("Invalid value for .df - must be pandas DataFrame.")
        self._df = new_df.copy()


    @property
    def polar_df(self):
        return(self._polar_df)
    @polar_df.setter
    def polar_df(self, new_df):
        if not isinstance(new_df, pd.DataFrame):
            raise ValueError("Invalid value for .polar_df - must be pandas DataFrame.")
        self._polar_df = new_df.copy()
    

    def _make_polar_df(self):
        """
        Transforms source data into a version ready for plotting and labeling.
        """
        # Matrix of tf-idf scores; rows = documents, columns = terms
        tfidf_df =  pd.DataFrame.from_dict(dict(zip(self.df[self.tfidf_col].index, self.df[self.tfidf_col].values))).T.fillna(0)
        
        # Reduce (by PCA) to the num_feats to use in visualization
        pca_df = pd.DataFrame(PCA(n_components=self.num_feats).fit_transform(tfidf_df), index=self.df[self.tfidf_col].index)
        # Normalize from [-1,1] to [0,1] scale
        pca_df = (pca_df + 1)/2

        # Assemble the version of the DF for plotting with polar coordinates
        polar_df = pd.DataFrame(pca_df.stack(level=0)).reset_index(level=1, names=['','pca_feat'])
        polar_df = polar_df.merge(self.df[self.cluster_col], how='left', left_index=True, right_index=True)
        polar_df.rename(columns={0:'pca_score', self.cluster_col:'cluster_id'}, inplace=True)
        for str_feat in ['pca_feat', 'cluster_id']:
            polar_df[str_feat] = polar_df[str_feat].astype(str)

        self.polar_df = polar_df


    def update_polar_df(self,
                        tfidf_col: str | None = None,
                        cluster_col: str | None = None,
                        num_feats: int | None = None):
        """ 
        Update the object's polar_df with new parameters.
        """
        if tfidf_col:
            self.tfidf_col = tfidf_col
        if cluster_col:
            self.cluster_col = cluster_col
        if num_feats:
            self.num_feats = num_feats

        self._make_polar_df()
        
    
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

        **kwargs : optional plotly keyword parameters
            > colorscale : str, plotly colorscale, default 'turbo_r'
            > colorscale_low : float, low end of the color spectrum, default 0.0
            > colorscale_high : float, high end of the color spectrum, default 1.0
            > horizontal_spacing : float, horizontal distance between subplots, default 0.0
            > vertical_spacing : float, vertical distance between subplots, default 0.1
            > paper_bgcolor : plotly color value, fill color outside the polar plot, default black 
            > plot_bgcolor : plotly color value, fill color inside the polar plot, default black
            > grid_color : plotly color value, line color of the boundary/axes of the polar plot, default grey
            > rounded_plot : bool, False makes plat polygonal, default True
            > fill_traces : bool, True fills the interior of each plot, default False
            > title : str, sets plot title, default "Plot of TF-IDF vectors"
            > showlegend : bool, controls the legend display, default True
            > template : str, plotly color template, default 'plotly_dark'
            > autosize : bool, automatically configure plot size, default True
            > width : int, sets plot width, default is autosize=True
            > height : int, sets plot height, default is autosize=True
                >> NOTE: autosize and width/height are mutually exclusive!
            > margin : dict, sets plot margins, optional
                >> EG: margin={'l':10,'r':10,'t':10,'b':10}
            > minreducedwidth : int, sets minimum (sub)plot width, optional
            > minreducedheight : int, sets minimum (sub)plot height, optional
    
        """

        if cluster_ids is None:
            cluster_ids = self.polar_df.cluster_id.unique().tolist()
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
                            vertical_spacing=kwargs.get('vertical_spacing',0.1))
            
        # Add trace for each cluster
        row = 1
        col = 1
        for cluster in sorted_cluster_ids:
            cluster_df = self.df.loc[self.df.cluster_id==cluster]
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

        if inplace:
            fig.show()
        else:
            return(fig)

# END OF PlotPolars class




###############################################
##    Highlight Document Text Differences    ##
###############################################

class DocDiff:
    def __init__(self,
                 df: pd.DataFrame,
                 ):
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