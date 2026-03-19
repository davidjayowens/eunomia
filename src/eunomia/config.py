# These variables define the boundaries of what the parameters of 
# legimongo.make_sample() consider "valid" values. Only bills from years 
# between MIN_YEAR and MAX_YEAR will be included in any sample; likewise
# for STATES.

# NOTE: "Invalid" values will be ignored when creating the sub-sample,
# rather than throwing exceptions or raising warnings! If no valid values
# are provided to .make_sample()'s parameters, all available values for the
# parameter will be included in sub-sample results (default behavior).

# Recommended behavior is to define the actual limits of the available
# data in MongoDB here, and then use the .make_sample() parameters to
# make specific sub-samples for clustering, rather than configuring a
# sub-sample's parameters here.

MIN_YEAR = 2009
MAX_YEAR = 2022
STATES = ['ak', 'al', 'ar', 'az', 'ca', 'co', 'ct', 'de', 'fl', 'ga', 
            'hi', 'ia', 'id', 'il', 'in', 'ks', 'ky', 'la', 'ma', 'md', 
            'me', 'mi', 'mn', 'mo', 'ms', 'mt', 'nc', 'nd', 'ne', 'nh', 
            'nj', 'nm', 'nv', 'ny', 'oh', 'ok', 'or', 'pa', 'ri', 'sc', 
            'sd', 'tn', 'tx', 'ut', 'va', 'vt', 'wa', 'wi', 'wv', 'wy']