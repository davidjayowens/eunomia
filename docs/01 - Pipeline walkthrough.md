# Data wrangling

The primary data source used in these examples was a bulk download of several zip files from [LegiScan](https://legiscan.com/). For users who wish to compile their own dataset of bills from LegiScan, `eunomia.legiscanner` (based on `pylegiscan` by Chris Polinquin) is included in the library to make scraping relatively simple.

## Data summary
After loading data into MongoDB using the steps below, my total collection includes 1,165,008 bills from across all 50 states, not including DC. Importantly, the bills are unevenly distributed - here's the breakdown of bill counts by state:  
```
AK :   4,431 (0.31%)  
AL :  14,698 (1.04%)  
AR :  13,878 (0.98%)  
AZ :  17,293 (1.22%)  
CA :  31,904 (2.26%)  
CO :   8,269 (0.59%)  
CT :  26,184 (1.85%)  
DE :   4,877 (0.35%)  
FL :  31,294 (2.22%)  
GA :  12,739 (0.90%)  
HI :  53,620 (3.80%)  
IA :  13,048 (0.92%)  
ID :   7,252 (0.51%)  
IL :  70,528 (5.00%)  
IN :  13,200 (0.93%)  
KS :   7,628 (0.54%)  
KY :  10,838 (0.77%)  
LA :  18,837 (1.33%)  
MA :  51,105 (3.62%)  
MD :  34,067 (2.41%)  
ME :  11,618 (0.82%)  
MI :  24,945 (1.77%)  
MN :  51,536 (3.65%)  
MO :  24,076 (1.71%)  
MS :  36,161 (2.56%)  
MT :   6,506 (0.46%)  
NC :  13,075 (0.93%)  
ND :   4,226 (0.30%)  
NE :   7,495 (0.53%)  
NH :  11,190 (0.79%)  
NJ :  57,556 (4.08%)  
NM :  12,126 (0.86%)  
NV :   6,368 (0.45%)  
NY : 124,330 (8.81%)  
OH :   7,284 (0.52%)  
OK :  46,094 (3.26%)  
OR :  17,350 (1.23%)  
PA :  27,924 (1.98%)  
RI :  24,053 (1.70%)  
SC :  11,176 (0.79%)  
SD :   6,111 (0.43%)  
TN :  43,762 (3.10%)  
TX :  32,790 (2.32%)  
UT :  10,057 (0.71%)  
VA :  28,804 (2.04%)  
VT :   7,508 (0.53%)  
WA :  24,072 (1.71%)  
WI :  12,858 (0.91%)  
WV :  23,677 (1.68%)  
WY :   4,590 (0.33%)  
```

In this walkthrough, we'll use a sample of bills from Georgia (2017-2020), but when building multi-state samples, the number of bills will be roughly proportional to the values above. So, for example, if creating a sample of bills from Oregon (17,350 bills) and Oklahoma (46,094 bills), with a sample size of 10%, there would be 1,735 Oregon bills and 4,609 Oklahoma bills in the final results. 

If applying additional filters to the sample, such as limiting by year, the number of bills from each state would be based on the total number of bills satisfying the filters, proportional by state.

## Prerequisites
After installing the `eunomia` package, you will need to make sure you have MongoDB installed and running. 

Note: These instructions are based on MacOS, so some of the instructions that follow may need to be modified for Windows/Linux users.

I used the free MongoDB community edition, which runs as a service via `brew`. If your setup is similar to mine, you can confirm the service is active by typing `brew services` in your terminal:

![brew services running mongodb](../imgs/brew_services.png)

Additionally, I would strongly recommend configuring Tika for offline mode - otherwise it attempts to connect to the Apache servers when processing documents, which can slow things down significantly. You'll first need to download a couple files - for details on this, refer to  [Tika's documentation](https://pypi.org/project/tika/) (specifically, the section "Airgap Environment Setup"). Once you've downloaded the required files, add this code snippet to the initialization of your script:
```python
import os
os.environ['TIKA_SERVER_JAR'] = "file:////tika_folder/tika-server-standard-3.1.0.jar"
```
You'll likely need to edit the actual path here, depending on your setup.

## Unpacking zip files, loading documents into MongoDB
For users who are scraping their own sample from LegiScan via the API, the below steps will need to be modified based on any local files you are using. If you're using `eunomia.legiscanner` to scrape data via the API, you can take the results, stored as a `pandas DataFrame`, and load them directly to MongoDB via the `Legiscan2Mongo.load_df()` method instead of using `.load_zips()` as demonstrated here.

```python
from eunomia.legimongo import Legiscan2Mongo

loader = Legiscan2Mongo(
    load_dir = load_folder, # Path to folder with the zips
    mongo_db = 'example_db', # Database to use in MongoDB
    mongo_coll = 'example_collection' # Collection to use in MongoDB
                        )
```
Optionally, the `Legiscan2Mongo` class can be initialized with the `verbose = True` parameter to print additional status messages as each file is being processed. Summaries of the processing results are returned on completion - these can be allowed to pass directly to stdout, stored and printed, or (my preference) logged.

I am a fan of logging, as a best practice. Although the details are not covered here, feel free to use my [logging utility](https://github.com/davidjayowens/djo/blob/main/src/djo/io.py).
```python
from djo.io import Log

log = Log(id = 'eunomia', 
          file_name = f'Eunomia data loading - Processing {state}',
          file_dir = log_folder,
          level = 'DEBUG',
          verbose = True)
```

Now we begin actually extracting files and processing bills:
```python
log.log(f"Loading data from zips in folder: {load_folder}")

load_result = loader.load_zips(bill_types=1)
log.log(load_result)
```

In addition to logging the results, the `Log(verbose=True)` parameter causes log messages to be printed to `stdout`. Here is an example from processing the zips in the Wyoming folder:
```
INFO >> Loading data from zips in folder: /Volumes/Eunomia/LegiScan/WY
Zipfiles: 100.000% done // Bills in zip: 100.000% done

INFO >> RESULTS:
15 zip files found in target directory: /Volumes/Eunomia/LegiScan/WY
5176 bills were processed out of 5176 identified
5176 records were successfully inserted into Mongo (100.000% success rate)
-------------------------------------------------------
Mongo database: eunomia
Mongo collection: prod1
-------------------------------------------------------
For more details, review: (self).loading_fails
```

A quick note on the bill_types parameter: there are many bill_type values in the LegiScan database, and most of them are not actual legislation. Some examples include:
* 1: 'Bill'
* 2: 'Resolution'
* 3: 'Concurrent Resolution'
* 4: 'Joint Resolution'
* 5: 'Joint Resolution Constitutional Amendment'
* 6: 'Executive Order'
* 7: 'Constitutional Amendment'
* 17: 'Initiative'
* 18: 'Petition'
* 19: 'Study Bill'
* 20: 'Initiative Petition'
* 21: 'Repeal Bill'
* 22: 'Remonstration'
* 23: 'Committee Bill'

To give you a sense of why I limit to only documents with bill_type=1, out of ~1.4 million documents in my collection, ~1.2 million of them are Bills (bill_type=1). Further, many of the documents with other bill_type values are procedural or ceremonial, such as declarations honoring local businesses and organizations. They also do not generally correspond to most model bill texts.


## Decoding bill texts
Once the raw documents are stored in your MongoDB collection, the actual document texts will need to be extracted from the base64 encoded strings.

```
Bills processed: 100.000% done // Decoded: 100.000% successful                
INFO >> RESULTS:
4777 bills were processed out of 4777 in current collection
4777 texts were successfully decoded and updated in Mongo (100.000% success rate)
-------------------------------------------------------
Mongo database: eunomia
Mongo collection: prod1
-------------------------------------------------------
For more details, review: (self).decoding_fails
```

Bill texts in the LegiScan database come in a variety of formats, and as a result, `eunomia.legimongo` has a relatively high number of dependencies, covered below.

| LegiScan mime_id | Document format |
| --- | --- |
| 1 | HTML |
| 2 | PDF |
| 4 | MS Word |
| 5 | RTF |

`eunomia.legimongo.Legiscan2Mongo` uses a combination of methods to convert the base64 strings into plain text. `decode_texts()` manages the general text extraction pipeline for all bills in the current collection, and it relies primarily on two subordinate functions to handle the heavy lifting:
- `extract_text(text, mime_id)` extracts document texts from the base64 format LegiScan provides.
    - For HTML files, a combination of [`BeautifulSoup`](https://beautiful-soup-4.readthedocs.io/en/latest/#) and regex (via `re`) are used.
    - For PDF and MS Word, [`tika-python`](https://pypi.org/project/tika/) is used - see [Prerequisites](#prerequisites) above for additional details on installation and setup requirements.
    - For RTF documents, the `rtf_to_text()` function from [`striprtf`](https://pypi.org/project/striprtf/) is used.
- `normalize_text(text)`


Tika is a powerful tool for extracting text from proprietary document formats, including MS Word and Adobe PDF. However, it does require some extra setup in order to make it useful in projects of this scale.

Reference: https://pypi.org/project/tika/ (see section "Airgap Environment Setup")  