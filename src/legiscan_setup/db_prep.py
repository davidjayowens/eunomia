"""
Unpacks data from the legiscan zip files, loads bill data into MongoDB.

Includes decoding bill text.

Input zips filename format:
LegiScan/{State}_{startYYYY}-{endYYYY}_{Session_Title}_{sessionID}.zip

Unzips to:
> {State}/{Session}/
    > bill/
        - {billID}.json
    > people/
        - {peopleID}.json
    > text/
        - {textID}.json
    > vote/
        - {voteID}.json
    - hash.md5
    - LICENSE
"""


import json
import random
from time import sleep
from pathlib import Path
import pymongo
import re
from zipfile import ZipFile, is_zipfile
import pandas as pd

from base64 import b64decode as b64d
import unicodedata
from bs4 import BeautifulSoup as bs
import tika
from tika import parser
from striprtf.striprtf import rtf_to_text


class Legiscan2Mongo:
    # These are specific to the current dataset
    MIN_YEAR = 2009
    MAX_YEAR = 2022
    STATES = ['ak', 'al', 'ar', 'az', 'ca', 'co', 'ct', 'de', 'fl', 'ga', 
              'hi', 'ia', 'id', 'il', 'in', 'ks', 'ky', 'la', 'ma', 'md', 
              'me', 'mi', 'mn', 'mo', 'ms', 'mt', 'nc', 'nd', 'ne', 'nh', 
              'nj', 'nm', 'nv', 'ny', 'oh', 'ok', 'or', 'pa', 'ri', 'sc', 
              'sd', 'tn', 'tx', 'ut', 'va', 'vt', 'wa', 'wi', 'wv', 'wy']
    

    def __init__(self,
                 mongo_db: str,
                 mongo_coll: str,
                 file_dir: str | Path | None = None,
                 verbose: bool = False) -> object:
        """
        Parameters
        ----------
        mongo_db : str
            Name of the Mongo database where the collection is saved.

        mongo_coll : str
            Name of the Mongo collection where the data is saved.

        file_dir : str or Path, optional
            Directory where Legiscan zipfiles are stored. 
            Only used by .load_zips().

        verbose : bool, default False
            If True, prints updates to stdout during processing.
        """
        
        # Connect to MongoDB and set target DB & collection
        # Store the parameter values for reference later
        self.MC = pymongo.MongoClient()

        self.mongo_db = mongo_db
        self.DB = self.MC[mongo_db]

        self.mongo_coll = mongo_coll
        self.COLL = self.DB[mongo_coll]

        # Store list of any samples created
        self.samples = []

        # Use current directory unless otherwise specified
        if file_dir is not None:
            self.DATA_FOLDER = Path(file_dir).resolve()
        else:
            self.DATA_FOLDER = None

        self.verbose = verbose


    def load_zips(self, 
                  bill_types: int | list[int] | None = None,
                  file_dir: str | Path | None = None,
                  verbose: bool | None = None):
        """
        Load all LegiScan zip files from target directory.

        Parameters
        ----------
        bill_types : int or list of ints, optional
            One or more Legiscan bill_type_id values:
            >  1: 'Bill'
            >  2: 'Resolution'
            >  3: 'Concurrent Resolution'
            >  4: 'Joint Resolution'
            >  5: 'Joint Resolution Constitutional Amendment'
            >  6: 'Executive Order'
            >  7: 'Constitutional Amendment'
            >  8: 'Memorial'
            >  9: 'Claim'
            > 10: 'Commendation'
            > 11: 'Committee Study Request'
            > 12: 'Joint Memorial'
            > 13: 'Proclamation'
            > 14: 'Study Request'
            > 15: 'Address'
            > 16: 'Concurrent Memorial'
            > 17: 'Initiative'
            > 18: 'Petition'
            > 19: 'Study Bill'
            > 20: 'Initiative Petition'
            > 21: 'Repeal Bill'
            > 22: 'Remonstration'
            > 23: 'Committee Bill'

        file_dir : str or Path, optional
            Directory where legiscan files are stored.

        verbose : bool, defaults to the value provided to init()
            If True, prints updates to stdout during processing.
        """
        if isinstance(bill_types, int) and (bill_types in range(1,24)):
            bill_type_filter = [str(bill_types)]
        elif isinstance(bill_types, list):
            bill_type_filter = [str(int(v)) for v in bill_types if int(v) in range(1,24)]
        else:
            bill_type_filter = None

        if (file_dir is None) and (self.DATA_FOLDER is None):
            raise ValueError("No directory provided.\nPlease provide a valid path to the file_dir parameter.")
        elif (file_dir is not None):
            self.DATA_FOLDER = Path(file_dir).resolve()
        # else self.DATA_FOLDER was configured on init, use that value

        if verbose is None:
            verbose = self.verbose

        allzips = [f for f in self.DATA_FOLDER.iterdir() if is_zipfile(f)]

        # Find bills in each .zip
        bill_json_pattern = re.compile(r".+/.+/bill/\d+.json")

        # Track which bills don't get added to Mongo
        self.loading_fails = []

        files_tot = len(allzips)
        files_cnt = 0
        bills_tot = 0
        bills_cnt = 0
        done_cnt = 0

        if verbose:
            print(f"{files_tot} files identified in {self.DATA_FOLDER.as_posix()}")

        for zip in allzips:
            files_cnt += 1

            if verbose:
                print(f"Inspecting zipfile: {zip}\n# {files_cnt} out of {files_tot}")

            with ZipFile(zip) as zf:
                manifest = [z for z in zf.namelist() if bill_json_pattern.match(z)]
                
                these_bills_tot = len(manifest)
                bills_tot += these_bills_tot
                these_bills_cnt = 0

                for bill in manifest:
                    bills_cnt += 1
                    these_bills_cnt += 1

                    if verbose:
                        print(f"Processing bill: {bill}\nZipfile {files_cnt} out of {files_tot} // Bill {these_bills_cnt} out of {these_bills_tot}\nCumulative: {done_cnt} successful out of {bills_cnt-1} attempted (this is {bills_cnt})")

                    try:
                        status = 'Loading bill to json_string'
                        json_string = json.loads(zf.read(bill))['bill']

                        if (bill_type_filter is not None) and (json_string["bill_type_id"] not in bill_type_filter):
                            if verbose:
                                print(f"Skipping invalid bill_type_id: {json_string['bill_type_id']}")
                            continue    # Skip processing

                        # load corresponding (encoded) text of initial version of the bill
                        status = "Identifying first available bill document"
                        first_doc_id = json_string["texts"][0]["doc_id"]
                        
                        status = 'Identifying path to file containing first bill document text'
                        first_doc_file = f"{Path(bill).parents[1] / 'text' / str(first_doc_id)}.json"
                        
                        status = 'Loading encoded bill text from file'
                        intro_text_body = json.loads(zf.read(first_doc_file))["text"]["doc"]

                        status = 'Constructing bill data dictionary'
                        temp_bill_data = {
                            "_id": json_string["bill_id"],
                            "bill_id": json_string["bill_id"],
                            "bill_type_id": json_string["bill_type_id"],

                            "state": json_string["state"],
                            "session_id": json_string["session"]["session_id"],
                            "session_yr_start": json_string["session"]["year_start"],
                            "session_yr_end": json_string["session"]["year_end"],
                            "session_special": json_string["session"]["special"],

                            "sponsor_ids": [d["people_id"] for d in json_string["sponsors"]],
                            "sponsor_parties": [d["party_id"] for d in json_string["sponsors"]],
                            
                            "title": json_string["title"],
                            "description": json_string["description"],
                            "doc_id": json_string["texts"][0]["doc_id"],
                            "doc_mime_id": json_string["texts"][0]["mime_id"],
                            "text_body_encoded": intro_text_body,
                            "text_body": None, # will be created in a later step
                        }

                        # Write bill record to MongoDB
                        status = "Writing record to Mongo"
                        self.COLL.update_one(temp_bill_data, {'$setOnInsert': temp_bill_data}, upsert=True)
                        
                        # Update counter
                        done_cnt += 1

                        if verbose:
                            print(f"Result: SUCCESS\n")
                        else:
                            print(f"Zipfiles: {(files_cnt/float(files_tot))*100:.3f}% done // Bills in zip: {(these_bills_cnt/float(these_bills_tot))*100:.3f}% done", end='                \r')

                    except Exception as e:
                        self.loading_fails.append((zip, bill, status, e))
                        
                        if verbose:
                            print(f"Result: FAILURE // Stopped at: {status} // Error: {e}\n")
        print()

        results =   f"""
                    RESULTS:
                    {files_tot} zip files found in target directory: {self.DATA_FOLDER}
                    {bills_cnt} bills were processed out of {bills_tot} identified
                    {done_cnt} records were successfully inserted into Mongo ({(done_cnt/float(bills_tot))*100:.3f}% success rate)
                    -------------------------------------------------------
                    Mongo database: {self.mongo_db}
                    Mongo collection: {self.mongo_coll}
                    -------------------------------------------------------
                    For more details, review: (self).loading_fails
                    """
        
        repl_patt = re.compile(r'\ {2,}|\t')
        results = re.sub(pattern=repl_patt, repl='', string=results).strip()

        return(results) 


    def save_loading_fails(self,
                           file_name: str):
        """
        Save the bills which failed to process as a .CSV file.

        NOTE: Does not include bills skipped due to bill_type_id filtering.

        Parameters
        ----------
        file_name : str or Path
            Name of the saved file.

        file_dir : str or Path, optional
            Location of the saved file.
        """
        if file_name.lower().endswith('.csv'):
            this_file_name = Path(file_name).resolve()
        else:
            this_file_name = Path(f'{file_name}.csv').resolve()

        # ((zip, bill, status, e))
        lf_df = pd.DataFrame(data=self.loading_fails, columns=['zip_file', 'bill_file', 'last_status', 'error'])

        lf_df.to_csv(this_file_name, index=False)


    def decode_texts(self, 
                     state: str | None = None,
                     undecoded_only: bool = False,
                     verbose: bool = False) -> str:
        """
        TODO: Need to make refinements to this for targetting
        individual bills.

        Decode the base64-encoded bill texts in the collection.

        Returns a status message like: 
        'X out of Y bills successfully decoded.'

        Parameters
        ----------
        state : str, optional
            Decode texts in the current collection for the given state only.

        undecoded_only : bool, default False
            Decode texts in the current collection that have not yet been decoded.

        verbose : bool, default False
            If True, prints updates to stdout during processing.
        """
        # Initiate Tika, for processing PDFs
        tika.initVM()

        # Track which bills don't get decoded successfully
        self.decoding_fails = []

        # Gather all bills in the current collection
        if undecoded_only and (state is not None):
            records = list(self.COLL.find({'state':state.upper(), 'text_body':None}))
        elif undecoded_only:
            records = list(self.COLL.find({'text_body':None}))
        elif state is not None:
            records = list(self.COLL.find({'state':state.upper()}))
        else:
            records = list(self.COLL.find({}))

        records_tot = len(records)
        records_cnt = 0
        done_cnt = 0

        if verbose:
            print(f"{records_tot} records identified in current collection")

        for record in records:
            records_cnt += 1

            try:
                id = record['_id']
                mime_id = record['doc_mime_id']
                encoded_text = record['text_body_encoded']
                if verbose:
                    print(f"Processing text of bill: {id}\nBill {records_cnt} out of {records_tot}\nCumulative: {done_cnt} successful out of {records_cnt-1} attempted (this is {records_cnt})")
                    print(f"Encoded text of bill {id}: {encoded_text[:50]}...")

                decoded_text = b64d(encoded_text)
                extracted_text = self.extract_text(decoded_text, mime_id)
                normalized_text = self.normalize_text(extracted_text)

                # Update the record
                self.COLL.update_one({"_id": id }, {"$set": {"text_body": normalized_text } } )

                done_cnt += 1

                if verbose:
                    print(f"Decoded text of bill {id}: {normalized_text[:50]}...\n")
                else:
                    print(f"Bills processed: {(records_cnt/float(records_tot))*100:.3f}% done // Decoded: {(done_cnt/float(records_cnt))*100:.3f}% successful", end='                \r')
            except Exception as e:
                self.decoding_fails.append((id, mime_id, e))

                if verbose:
                    print(f"Unable to decode bill {id}: {e}\n")
        
        print()

        results =   f"""
                    RESULTS:
                    {records_cnt} bills were processed out of {records_tot} in current collection
                    {done_cnt} texts were successfully decoded and updated in Mongo ({(done_cnt/float(records_tot))*100:.3f}% success rate)
                    -------------------------------------------------------
                    Mongo database: {self.mongo_db}
                    Mongo collection: {self.mongo_coll}
                    -------------------------------------------------------
                    For more details, review: (self).decoding_fails
                    """
        
        repl_patt = re.compile(r'\ {2,}|\t')
        results = re.sub(pattern=repl_patt, repl='', string=results).strip()

        return(results) 


    def save_decoding_fails(self,
                            file_name: str):
        """
        Save the bills which failed to process as a .CSV file.

        NOTE: Does not include bills skipped due to bill_type_id filtering.

        Parameters
        ----------
        file_name : str or Path
            Name of the saved file.

        file_dir : str or Path, optional
            Location of the saved file.
        """
        if file_name.lower().endswith('.csv'):
            this_file_name = Path(file_name).resolve()
        else:
            this_file_name = Path(f'{file_name}.csv').resolve()

        # ((zip, bill, status, e))
        df_df = pd.DataFrame(data=self.decoding_fails, columns=['bill_id', 'mime_id', 'error'])

        df_df.to_csv(this_file_name, index=False)

   
    @staticmethod
    def extract_text(text, mime_id):
        """
        Extract text from decoded string, based on MIME type
        """
        if mime_id == 1:
            # HTML // mime_id: 1
            soup = bs(text, features="lxml")

            for data in soup(['style', 'script', 'class', 'id', 'name']):
                # Remove all tags and scripts
                data.decompose()

            decoded_text = ' '.join(soup.stripped_strings)

            # Clean up any HTML comments
            html_comments_patt = re.compile(r'(<!--.+-->)')
            decoded_text = re.sub(html_comments_patt, '', decoded_text)    

        elif mime_id == 2:
            # PDF // mime_id: 2
            parsed = parser.from_buffer(text)
            decoded_text = parsed['content']

        elif mime_id == 4:
            # MS Word // mime_id: 4
            parsed = parser.from_buffer(text)
            decoded_text = parsed['content']

        elif mime_id == 5:
            # RTF // mime_id: 5
            decoded_text = rtf_to_text(text.decode())

        else:
            return(None)

        flat_text = Legiscan2Mongo._flatten(decoded_text)
        return(flat_text)

    
    @staticmethod
    def _flatten(text):
        """
        Flatten text into a single line; reduce excess whitespace.
        """
        repl_patt = re.compile(r'\s+')
        results = re.sub(pattern=repl_patt, repl=' ', string=text).replace("\xa0", " ").strip()

        return(results)

    
    @staticmethod
    def normalize_text(text):
        """
        NFKD form: "Normalization Form Compatibility Decomposition"

        Improves text matching by removing character accents/decorators, 
        reducing each letter to its base form (so 'é' -> 'e', etc).

        Note that this may result in some non-normalizeable characters
        being removed from the final text.
        """

        try:
            nfkd_form = unicodedata.normalize('NFKD', text)
            only_ascii = nfkd_form.encode('ASCII', 'ignore').decode('utf-8', 'ignore')
            finalresult = only_ascii.lower()

            return(finalresult)
        except:
            return(None)
        

    def make_sample(self,
                    new_coll_name: str,
                    samp_size: float = 0.01,
                    states: str | list[str] | None = None,
                    years: int | str | list[int] | list[str] | None = None,
                    bill_types: int | str | list[int] | list[str] | None = None,
                    verbose: bool | None = None):
        """
        Make a new sample collection in MongoDB to use in 
        clustering w/ BillCluster.
        
        Parameters
        ----------
        new_coll_name : str
            Name for the new MongoDB collection to store the sample.

        samp_size : float, default 0.01
            Sample size - the portion of available bills to use,
            after filtering by states and years.

        states : str or list of them, optional
            State or states to include in the sample.

        years : int, str, or list of them, optional
            Year or years to include in the sample.

        bill_types : int or list of them, optional
            One or more valid bill_type values (1 to 23) to include the sample.

        verbose : bool, default same as init
            If True, print messages while processing.
        """
        if verbose is None:
            verbose = self.verbose

        # Store sample size
        if isinstance(samp_size, float) and (0.0 <= samp_size) and (samp_size <= 1.0):
            self.samp_size = samp_size
        else:
            raise ValueError(f"Invalid parameter: {samp_size=} (must be 0.0 - 1.0)")
        
        if verbose:
            print(f"Sample size: {self.samp_size}")

        # Build list of states to filter on
        if isinstance(states, str):
            states = [states]
        
        if isinstance(states, list) and (len(states) > 0):
            self.states = [state.strip().upper()[:2] for state in states if state.strip().lower()[:2] in self.STATES]
            
            if len(self.states) == 0:
                self.states = None
        else:
            self.states = None

        if verbose:
            print(f"States list: {self.states}")

        # Build list of years to filter on (all int)
        if isinstance(years, (int, str, float)):
            years = [int(years)]

        if isinstance(years, list) and (len(years) > 0):
            self.years = [int(year) for year in years if self.MIN_YEAR <= int(year) <= self.MAX_YEAR]

            if len(self.years) == 0:
                self.years = None
        else:
            self.years = None

        if verbose:
            print(f"Years list: {self.years}")

        # Building list of bill types
        if isinstance(bill_types, (int, str)):
            bill_types = [str(bill_types)]

        if isinstance(bill_types, list) and (len(bill_types) > 0):
            self.bill_types = [str(bill_type) for bill_type in bill_types if 1 <= int(bill_type) <= 23]

            if len(self.bill_types) == 0:
                self.bill_types = None
        else:
            self.bill_types = None

        if verbose:
            print(f"Bill types list: {self.bill_types}")

        if verbose:
            print("Creating new sample...")

        # Find all bills that match the scope
        pline_dict_01 = {'$match': {}}
        if self.states:
            pline_dict_01['$match']['state'] = {'$in': self.states}
        if self.years:
            pline_dict_01['$match']['session_yr_start'] = {'$in': self.years}
        if self.bill_types:
            pline_dict_01['$match']['bill_type_id'] = {'$in': self.bill_types}
   
        pline = [pline_dict_01]
        sample = list(self.COLL.aggregate(pline))
        num_bills = len(sample)

        if verbose:
            print(f"Total number of bills in target scope: {num_bills}")
            print(f"Estimated number of bills in sample: {int(num_bills * self.samp_size)}")

        # Get in-scope bills by state
        pline_dict_02 = {"$group": {"_id": "$state", "bill_ids": {"$addToSet": "$bill_id"}}}
        pline = [
            pline_dict_01,  # 01: Filter to sample scope, then
            pline_dict_02   # 02: Get available bills by state
        ]
        bills_by_state = {i['_id']:i['bill_ids'] for i in self.COLL.aggregate(pline)}
  
        bill_counts_by_state = {k:len(v) for k,v in bills_by_state.items()}
        samp_counts_by_state = {k:int((v * self.samp_size)+1) for k,v in bill_counts_by_state.items()}
        
        if verbose:
            print(f"Sample sizes by state:\n{samp_counts_by_state}")

        # Pull a random sample of bill IDs from each state
        rand_samp = {}
        rand_samp_flat = []
        for st in bills_by_state:
            rand_samp[st] = random.sample(bills_by_state[st], samp_counts_by_state[st])
            rand_samp_flat.extend(rand_samp[st])

        # Grab the records for all bills in the new sample from the master collection
        if verbose:
            print("Gathering sampled bills from primary Mongo collection...")

        pline_dict_03 = {"$match": {"bill_id": {"$in": rand_samp_flat}}}
        pline = [
            pline_dict_03
        ]
        sample = list(self.COLL.aggregate(pline))
        
        # Create new collection to store sample in Mongo
        if new_coll_name not in self.samples:
            self.samples.append(new_coll_name)

        samp_coll = self.DB[new_coll_name]
        doc_count = samp_coll.count_documents({})

        if doc_count > 0:
            print(f"Dropping existing collection: {new_coll_name}")
            samp_coll.drop()

        result = samp_coll.insert_many(sample, ordered=False)

        fails = [record['bill_id'] for record in sample if record['bill_id'] not in result.inserted_ids]
        if verbose:
            print(f"Successfully inserted: {len(sample) - len(fails)}")
            print(f"Failed to insert: {len(fails)}")

        # Assign results to object property for validation
        self.sample_raw = sample 



    # END OF Legiscan2Mongo class


