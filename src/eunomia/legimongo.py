from __future__ import annotations

import json
import random
from pathlib import Path
import re
from zipfile import ZipFile, is_zipfile
from time import sleep

import pymongo
import pandas as pd

from base64 import b64decode as b64d
import unicodedata
from bs4 import BeautifulSoup as bs
import tika
from tika import parser
from striprtf.striprtf import rtf_to_text

from eunomia.config import MIN_YEAR, MAX_YEAR, STATES

import logging
log = logging.getLogger(__name__)

def _log(msg:str, verbose:bool=False) -> None:
    """ Log msg at debug level and optionally print to stdout. """
    log.debug(msg, stacklevel=2)
    if verbose:
        print(msg)

class LegimongoError(Exception):
    def __init__(self, msg):
        log.error("Eunomia | LegimongoError exception encountered:\n" + msg, stacklevel=2)
        super().__init__(msg)


class Legiscan2Mongo:
    __slots__ = (
        'mongo_db', 
        'mongo_coll',
        'verbose',
        'Mongo',
        '_df',
        '_data_dir',
        'loading_fails',
        'decoding_fails'
        )

    def __init__(self,
                 mongo_db: str,
                 mongo_coll: str,
                 verbose: bool = False) -> object:
        """
        Class to load LegiScan data into MongoDB and decode bill texts.

        Parameters
        ----------
        mongo_db : str
            Name of the Mongo database where the collection will be saved.

        mongo_coll : str
            Name of the Mongo collection where the data will be saved.
            
        verbose : bool, default False
            If True, prints updates to stdout during processing.
        """
        self.verbose = verbose
        
        # Connect to MongoDB and set target DB & collection
        self.mongo_db = mongo_db
        self.mongo_coll = mongo_coll
        self.Mongo = MongoDF(db=mongo_db, coll=mongo_coll, local_mem=False, verbose=verbose)

        self.loading_fails = []    # Stores details of bills which could not be loaded
        self.decoding_fails = []   # Stores details of bills which could not be decoded

        _log(f"Successfully initialized:\n{self}", self.verbose)

    # END OF __init__

    ##########################################
    ##    Legiscan2Mongo Special Methods    ##
    ##########################################

    def __repr__(self):
        return(f"Legiscan2Mongo(mongo_db={self.mongo_db}, mongo_coll={self.mongo_coll}, verbose={self.verbose})")

    def __str__(self):
        return( "Legiscan2Mongo\n==============\n"
               f"mongo_db = {self.mongo_db}\n"
               f"mongo_coll = {self.mongo_coll}\n"
               f"verbose = {self.verbose}" 
              )
    

    #####################################
    ##    Legiscan2Mongo Attributes:   ## 
    ##        Getters & Setters        ##
    #####################################

    @property
    def df(self) -> pd.DataFrame:
        return(self._df)
    @df.setter
    def df(self, new_data):
        if not isinstance(new_data, pd.DataFrame):
            raise LegimongoError(f"Invalid object of type: {type(new_data)}\nMust be pandas DataFrame.")
        
        _log(f"Updating df with new DataFrame containing columns: {new_data.columns.tolist()}", self.verbose)
        self._df = new_data.copy()


    @property
    def data_dir(self) -> Path:
        return(self._data_dir)
    @data_dir.setter
    def data_dir(self, new_dir):
        try:
            new_path = Path(new_dir).resolve(strict=True)
        except OSError:
            raise LegimongoError(f"Unable to resolve folder location.\nPlease confirm path is correct and folder exists:\n{new_dir}")
        
        _log(f"Updating data_dir to folder: {new_path.as_posix()}", self.verbose)
        self._data_dir = new_path


    ##########################################
    ##    Legiscan2Mongo Primary Methods    ##
    ##########################################

    def load_df(self,
                df: pd.DataFrame,
                subset: str | list[str] | None = None,
                verbose: bool | None = None):
        """
        Load data from a pandas DataFrame into MongoDB.

        Parameters
        ----------
        df : pandas DataFrame
            Data to be loaded into the MongoDB collection.

        subset : str or list of them, optional
            One or more columns to use as a subset of the provided DataFrame
            when loading into MongoDB.
        
        verbose : bool, optional
            If True, prints updates to stdout during processing. If not
            provided, uses the value configured on init.

        """
        if verbose is None:
            verbose = self.verbose

        if isinstance(subset, str):
            subset = [subset]
        if isinstance(subset, list):
            self.df = df[subset]
        else:
            self.df = df

        _log(f"Updating records from pandas DataFrame into current collection: {self.mongo_db}.{self.mongo_coll}", verbose)
        self.Mongo._update_mongo(self.df)
        _log("MongoDB updates complete.", verbose)


    def load_zips(self, 
                  file_dir: str | Path,
                  bill_types: int | str | list[int|str] | None = None,
                  verbose: bool | None = None):
        """
        Load all LegiScan zip files in target directory into MongoDB.

        Parameters
        ----------
        file_dir : str or Path
            Directory where LegiScan zip files are stored.
        
        bill_types : int or list of ints, optional
            One or more LegiScan bill_type_id values
            NOTE: See legiscanner.BILL_TYPES for a dictionary of bill_type_id
                    values and their meanings.

        verbose : bool, defaults to the value provided to init()
            If True, prints updates during processing.
        """
        if verbose is None:
            verbose = self.verbose
        
        if isinstance(bill_types, (int,str)):
            bill_types = [bill_types]
        
        if isinstance(bill_types, list):
            bill_type_filter = [str(int(v)) for v in bill_types if int(v) in range(1,24)]
        else:
            bill_type_filter = None

        # Resolve input folder
        self.data_dir = file_dir

        allzips = [f for f in self.data_dir.iterdir() if is_zipfile(f)]

        # Find bills in each .zip
        bill_json_pattern = re.compile(r".+/.+/bill/\d+.json")

        # Keep count of current progress
        files_tot = len(allzips)
        files_cnt = 0
        bills_tot = 0
        bills_cnt = 0
        done_cnt = 0

        _log(f"{files_tot} files identified in {self.data_dir.as_posix()}", verbose)
        _log(f"Updating records from zip files into current collection: {self.mongo_db}.{self.mongo_coll}", verbose)
        for zip in allzips:
            files_cnt += 1

            _log(f"Inspecting zipfile: {zip}\n# {files_cnt} out of {files_tot}", verbose)

            with ZipFile(zip) as zf:
                manifest = [z for z in zf.namelist() if bill_json_pattern.match(z)]
                
                these_bills_tot = len(manifest)
                bills_tot += these_bills_tot
                these_bills_cnt = 0

                for bill in manifest:
                    bills_cnt += 1
                    these_bills_cnt += 1

                    _log(f"Processing bill: {bill}\n"
                         f"Zipfile {files_cnt} out of {files_tot} // "
                         f"Bill {these_bills_cnt} out of {these_bills_tot}\n"
                         f"Cumulative: {done_cnt} successful out of {bills_cnt-1} attempted (this is {bills_cnt})", verbose)

                    try:
                        # The status variable is included in the error message 
                        # if an exception is raised during processing
                        status = 'Loading bill to json_string'
                        json_string = json.loads(zf.read(bill))['bill']

                        if (bill_type_filter is not None) and (json_string["bill_type_id"] not in bill_type_filter):
                            _log(f"Skipping invalid bill_type_id: {json_string['bill_type_id']}", verbose)
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
                        self.Mongo._update_mongo(temp_bill_data)
                        
                        # Update counter
                        done_cnt += 1

                        if verbose:
                            _log(f"Result: SUCCESS", verbose)
                        else:
                            # Continuously updates status message on same line
                            print(f"Zipfiles: {(files_cnt/float(files_tot))*100:.3f}% done // "
                                  f"Bills in zip: {(these_bills_cnt/float(these_bills_tot))*100:.3f}% done", 
                                  end='                \r')

                    except Exception as e:
                        self.loading_fails.append((zip, bill, status, e))
                        
                        _log(f"Result: FAILURE // Stopped at: {status} // Error: {e}", verbose)

        results =   f"""
                    RESULTS:
                    {files_tot} zip files found in target directory: {self.data_dir.as_posix()}
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

        _log(results, verbose)


    def decode_texts(self, 
                     normalize: bool = True,
                     state: str | None = None,
                     undecoded_only: bool = True,
                     verbose: bool | None = None) -> str:
        """
        Decode the base64-encoded bill texts in the collection.

        Returns a status message like: 
        'X out of Y bills successfully decoded.'

        Parameters
        ----------
        normalize : bool, default True
            If True, applies Normalization Form Compatibility Decomposition (NFKD)
            to decoded texts and converts all characters to lowercase, 
            to improve text matching/clustering potential;
            if False, decoded texts are unmodified.

        state : str, optional
            Decode texts in the current collection for the given state only.

        undecoded_only : bool, default True
            Decode texts in the current collection that have not yet been decoded.

        verbose : bool, default False
            If True, prints updates to stdout during processing.
        """
        if verbose is None:
            verbose = self.verbose

        # Initiate Tika, for processing PDFs
        tika.initVM()
        sleep(5)    # Give Tika time to warm up

        # Gather all bills in the current collection
        search = {}
        if undecoded_only:
            search['text_body'] = None
        if state:
            search['state'] = state.upper()
        
        records_df = self.Mongo.get_records(search, inplace=False)[['_id', 'doc_mime_id', 'text_body_encoded']]

        records_tot = records_df.shape[0]
        records_cnt = 0
        done_cnt = 0

        _log(f"{records_tot} records identified in current collection", verbose)

        for _,record in records_df.iterrows():
            records_cnt += 1

            try:
                id = record['_id']
                mime_id = record['doc_mime_id']
                encoded_text = record['text_body_encoded']
                
                _log(f"Processing text of bill: {id}\nBill {records_cnt} out of {records_tot}\nCumulative: {done_cnt} successful out of {records_cnt-1} attempted (this is {records_cnt})", verbose)
                _log(f"Encoded text of bill {id}: {encoded_text[:50]}...", verbose)

                decoded_text = b64d(encoded_text)
                extracted_text = self.extract_text(decoded_text, mime_id)
                if normalize:
                    extracted_text = self.normalize_text(extracted_text)

                # Update the record
                # TODO: Test using MongoDF._update_mongo() here
                self.Mongo._COLL.update_one({"_id": id}, {"$set": {"text_body": extracted_text}})

                done_cnt += 1

                if verbose:
                    _log(f"Decoded text of bill {id}: {extracted_text[:50]}...\n", verbose)
                else:
                    # Continuously updates on same line
                    print(f"Bills processed: {(records_cnt/float(records_tot))*100:.3f}% done // "
                          f"Decoded: {(done_cnt/float(records_cnt))*100:.3f}% successful", 
                          end='                \r')
            except Exception as e:
                self.decoding_fails.append((id, mime_id, e))
                _log(f"Unable to decode bill {id}: {e}", verbose)

        # All texts processed
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
        # Remove extra whitespace
        repl_patt = re.compile(r'\ {2,}|\t')
        results = re.sub(pattern=repl_patt, repl='', string=results).strip()

        _log(results, verbose)

   
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
        
# END OF Legiscan2Mongo class




class MongoDF:
    __slots__ = (
        'local_mem',
        'df_in',
        'verbose',
        '_MC',
        '_db',
        '_DB',
        '_coll',
        '_COLL',
        '_df'
    )

    def __init__(self,
                 mongo_db: str,
                 mongo_coll: str,
                 local_mem: bool = False,
                 df: pd.DataFrame | None = None,
                 verbose: bool = False) -> object:
        """
        Creates an object to manage and modify MongoDB data.

        Parameters
        ----------
        mongo_db : str
            Name of the Mongo database where the collection is saved.

        mongo_coll : str
            Name of the Mongo collection where the data is saved.

        local_mem : bool, default False
            Copy collection into local memory, stored as a pandas DataFrame;
            can be slow to initialize, but makes searches faster.

        df : pandas DataFrame, optional
            Data to add to the MongoDB collection. If the collection
            already contains data, records provided to df parameter will be
            inserted/updated in the collection.

        verbose : bool, default False
            If True, prints updates to stdout during processing.
        """
        self.verbose = verbose
        self.local_mem = local_mem
        
        # Connect to MongoDB
        self._MC = pymongo.MongoClient()
        self.use_coll(mongo_db, mongo_coll)
        
        self.df_in = isinstance(df, pd.DataFrame)
        # If new data provided
        if self.df_in:
            # Insert/update new data
            self._update_mongo(df)
            
    # END OF __init__

    ###################################
    ##    MongoDF Special Methods    ##
    ###################################

    def __repr__(self):
        return(f"MongoDF(mongo_db={self.mongo_db}, mongo_coll={self.mongo_coll}, local_mem={self.local_mem}, "
               f"df={'<pd.DataFrame>' if self.df_in else 'None'}, verbose={self.verbose})")
        
    def __str__(self):
        return(f"MongoDF\n=======\n"
               f"mongo_db = {self.mongo_db}\n"
               f"mongo_coll = {self.mongo_coll}\n"
               f"local_mem = {self.local_mem}\n"
               f"df = {'<pd.DataFrame>' if self.df_in else 'None'}\n"
               f"verbose = {self.verbose})")


    ###############################
    ##    MongoDF Attributes:    ##
    ##     Getters & Setters     ##
    ###############################

    @property  
    def mongo_db(self) -> str:
        return(self._db)
    @mongo_db.setter
    def mongo_db(self, new_db):
        self._db = str(new_db)
        self._DB = self._MC[self._db]


    @property
    def mongo_coll(self) -> str:
        return(self._coll)
    @mongo_coll.setter
    def mongo_coll(self, new_coll):
        self._coll = str(new_coll)
        self._COLL = self._DB[self._coll]


    @property
    def df(self) -> pd.DataFrame:
        return(self._df)
    @df.setter 
    def df(self, new_data):
        if not isinstance(new_data, pd.DataFrame):
            raise LegimongoError(f"Invalid object of type: {type(new_data)}\nMust be pandas DataFrame.")
        
        _log(f"Updating df with new DataFrame containing columns: {new_data.columns.tolist()}", self.verbose)
        self._df = new_data.copy()
    

    ###############################
    ##    MongoDF Attributes:    ##
    ##      Updater Methods      ##
    ###############################

    def use_coll(   self,
                    mongo_db: str | None = None,
                    mongo_coll: str | None = None):
        """
        Use the specified MongoDB collection. Updates local DataFrame (object.df)
        with collection data.

        Parameters
        ----------
        new_db : str, optional
            Name of the Mongo database where the new collection is saved.

        new_coll : str, optional
            Name of the Mongo collection where the new data is saved.

        """
        if mongo_db:
            self.mongo_db = mongo_db
        if mongo_coll:
            self.mongo_coll = mongo_coll
        
        if self.local_mem:
            self.get_records(inplace=True)
    # Aliases:
    use_collection = use_coll
    set_coll = use_coll
    set_collection = use_coll
    

    def _update_mongo(  self,
                        data: dict | list[dict] | pd.DataFrame):
        """
        Inserts/updates data in the MongoDB collection.

        A dict can only be used to add a single record at a time,
        but a DataFrame can add one or multiple.
        """
        _log(f"Inserting/updating data in MongoDB collection: {self.mongo_coll}", self.verbose)

        def _upsert(record: dict):
            """ Add/update a single record in MongoDB """
            self._COLL.update_one(record, {'$setOnInsert': record}, upsert=True)

        try:
            if isinstance(data, dict):
                _upsert(data)

            elif isinstance(data, list):
                for d in list:
                    if isinstance(d, dict):
                        _upsert(d)
                    else:
                        raise LegimongoError(f"Invalid data of type: {type(data)}\nLists must only contain records of type dict.")
                            
            elif isinstance(data, pd.DataFrame):
                for d in data.to_dict(orient='records'):
                    _upsert(d)
            
            else:
                raise LegimongoError(f"Invalid data of type: {type(data)}\nMust be a pandas DataFrame, dict, or list of dicts.")
            
            # Optionally, sync updated collection back into self.df
            if self.local_mem:
                self.get_records(inplace=True)

        except Exception as e:
            raise LegimongoError(f"Error while updating MongoDB collection:\n{e}")
        

    ###################################
    ##    MongoDF Primary Methods    ##
    ###################################

    def get_records(self,
                    search: dict | list[dict] | None = None,
                    pandas: bool = True,
                    inplace: bool | None = None,
                    mongo_db: str | None = None,
                    mongo_coll: str | None = None) -> pd.DataFrame:
        """
        Returns bills matching either a single search query (dict) or
        an aggregated search pipeline (list of dicts).

        Parameters
        ----------
        search : dict or list[dict], optional
            If a dict is provided, uses MongoDB's find() method; if a list of
            dicts is provided, uses MongoDB's aggregate() method. If no search 
            terms are provided (default), returns all available
            documents in current collection.

        pandas : bool, default True
            If True, convert results to a pandas DataFrame;
            if False, return results in raw form.
            NOTE: If inplace=True, this value will always be treated as True.
            
        inplace : bool, optional
            If True, updates the object's .df property with the results;
            if False, returns results directly. By default, uses the same
            value as the local_mem parameter provided during init.

        mongo_db : str, optional
            If provided, overrides the currently selected MongoDB database.

        mongo_coll : str, optional
            If provided, overrides the currently selected MongoDB collection.
            
        """
        if search is None:
            search = {}
        if inplace is None:
            inplace = self.local_mem

        if mongo_db:
            this_db = self._MC[mongo_db]
        else:
            this_db = self._DB

        if mongo_coll:
            this_coll = this_db[mongo_coll]
        else:
            this_coll = this_db[self.mongo_coll]

        try:
            if isinstance(search, dict):
                results = this_coll.find(search)
            elif isinstance(search, list):
                results = this_coll.aggregate(search)
            else:
                raise LegimongoError(f"Invalid search of type: {type(search)}\nMust be either dict (find) or list of dicts (aggregate)")

            if pandas or inplace:
                results = pd.DataFrame(list(results))
            if inplace:
                self.df = results
            else:
                return(results)
            
        except Exception as e:
            raise LegimongoError(f"Unable to execute search query.\n{e}")

        
    def count_records(self,
                      search: dict | None = None,
                      mongo_db: str | None = None,
                      mongo_coll: str | None = None) -> int:
        """
        Returns count of bills matching the search query.

        Parameters
        ----------
        search : dict, optional
            Dictionary of search parameters; if no search terms are provided,
            counts all documents in current collection.

        mongo_db : str, optional
            If provided, overrides the currently selected MongoDB database.

        mongo_coll : str, optional
            If provided, overrides the currently selected MongoDB collection.

        """
        if search is None:
            search = {}

        if mongo_db:
            this_db = self._MC[mongo_db]
        else:
            this_db = self._DB

        if mongo_coll:
            this_coll = this_db[mongo_coll]
        else:
            this_coll = this_db[self.mongo_coll]
            
        try:
            if not isinstance(search, dict):
                raise LegimongoError(f"Invalid search of type: {type(search)}\nMust be a dict.")
                
            count = self._COLL.count_documents(search)
            return(count)
            
        except Exception as e:
            raise LegimongoError(f"Unable to execute search query.\n{e}")


    def make_subsample( self,
                        new_coll: str,
                        new_db: str | None = None,
                        samp_size: float = 0.01,
                        states: str | list[str] | None = None,
                        years: int | str | list[int|str] | None = None,
                        bill_types: int | str | list[int|str] | None = None,
                        append: bool = True,
                        update_self: bool = False,
                        verbose: bool | None = None):
        """
        Make a new sample in MongoDB as a subsample of the current collection. 
        Results are stratified by state, based on number of bills available in the 
        targeted scope.
        
        Parameters
        ----------
        new_coll : str
            Name for the new MongoDB collection to store the sample.

        new_db : str, optional
            If provided, the MongoDB database where sample will be stored;
            by default, uses currently selected database.

        samp_size : float, default 0.01
            Sample size - the portion of available bills to use,
            after filtering by states and years.

        states : str or list of them, optional
            State or states to include in the sample.

        years : int, str, or list of them, optional
            Year or years to include in the sample.

        bill_types : int or list of them, optional
            One or more valid bill_type values (1 to 23) to include the sample.

        append : bool, default True
            If new_coll already exists in the current database and 
            append=True, records will be inserted/updated in the existing collection;
            if False, the existing collection will be dropped and replaced.
            
        update_self: bool, default False
            If True, immediately switch to the new collection and replace 
            existing data in local memory (if local_mem=True) with the new sample; 
            if False, the new collection is created without modifying the
            current parameters or data.

        verbose : bool, defaults to the value provided to init()
            If True, prints updates during processing.

        """
        if verbose is None:
            verbose = self.verbose

        # Validate samp_size
        if not (isinstance(samp_size, float) and (0.0 <= samp_size <= 1.0)):
            raise LegimongoError(f"Invalid parameter: {samp_size=}\nMust be float between 0.0 and 1.0.")
        
        # Build list of states to filter on
        if isinstance(states, str):
            states = [states]
        if isinstance(states, list):
            states = [state.strip().upper()[:2] for state in states 
                        if state.strip().lower()[:2] in STATES]
            if len(states) == 0:
                states = None
        else:
            states = None

        # Build list of years to filter on (all int)
        if isinstance(years, (int, str, float)):
            years = [years]
        if isinstance(years, list) and (len(years) > 0):
            years = [int(year) for year in years 
                        if (MIN_YEAR <= int(year) <= MAX_YEAR)]
            if len(years) == 0:
                years = None
        else:
            years = None

        # Building list of bill types (all str)
        if isinstance(bill_types, (int, str)):
            bill_types = [bill_types]
        if isinstance(bill_types, list):
            bill_types = [str(bill_type) for bill_type in bill_types 
                            if (1 <= int(bill_type) <= 23)]
            if len(bill_types) == 0:
                bill_types = None
        else:
            bill_types = None

        # Log all params for new sample after validation
        _log(f"CREATING NEW SAMPLE\n===================\n"
             f"DB: {new_db if new_db else self.mongo_db}\n"
             f"Collection: {new_coll}\n"
             f"Sample size: {samp_size}\n"
             f"States: {states if states else 'ALL'}\n"
             f"Years: {years if years else 'ALL'}\n"
             f"Bill types: {bill_types if bill_types else 'ALL'}\n"
             f"Append: {append}",
             verbose)

        # Find bills that match the scope in current collection
        query = {}
        if states:
            query['state'] = {'$in': states}
        if years:
            query['session_yr_start'] = {'$in': years}
        if bill_types:
            query['bill_type_id'] = {'$in': bill_types}
   
        pline_dict_01 = {'$match': query}
        num_bills = self.count_records(query)

        _log(f"Total number of bills in target scope: {num_bills}\n"
             f"Estimated number of bills in sample: {int(num_bills * samp_size)}",
             verbose)

        # Get in-scope bills by state
        pline_dict_02 = {"$group": {"_id": "$state", "bill_ids": {"$addToSet": "$bill_id"}}}
        pline = [
            pline_dict_01,  # 01: Filter to sample scope, then
            pline_dict_02   # 02: Get available bills by state
        ]
        bills_by_state = {st['_id']:st['bill_ids'] for st in self.get_records(pline, pandas=False, inplace=False)}
  
        alpha_states = sorted(list(bills_by_state.keys()))
        bill_counts_by_state = {st:len(bills_by_state[st]) for st in alpha_states}
        samp_counts_by_state = {k:int((v * self.samp_size)+1) for k,v in bill_counts_by_state.items()}
        
        _log(f"Sample sizes by state:\n{samp_counts_by_state}", verbose)

        # Pull a random sample of bill IDs from each state
        rand_samp = {}
        rand_samp_flat = []
        for st in alpha_states:
            rand_samp[st] = random.sample(bills_by_state[st], samp_counts_by_state[st])
            rand_samp_flat.extend(rand_samp[st])

        # Potentially drop existing collection
        doc_count = self.count_records(mongo_db=new_db, mongo_coll=new_coll)
        if not append and (doc_count > 0):
            tar_db = new_db if new_db else self.mongo_db
            _log("Dropping existing collection\n============================\n"
                f"DB: {tar_db}\n"
                f"Collection: {new_coll}\n"
                f"Existing record count: {doc_count}")
            try:
                self._MC[tar_db][new_coll].drop()
                _log("Operation successful.", verbose)
            except Exception as e:
                raise LegimongoError(f"Unable to drop collection.\n{e}")

        # Copy records directly into new location
        pline_dict_03 = {"$match": {"bill_id": {"$in": rand_samp_flat}}}
        if new_db:
            pline_dict_04 = {"$out": {"db": new_db, "coll": new_coll}}
        else:
            pline_dict_04 = {"$out": new_coll}

        pline = [
            pline_dict_03,  # Matches bills in the sample in current collection
            pline_dict_04   # Copies bills to the new collection directly
        ]
        self._COLL.aggregate(pline)

        # Update object data; if self.local_mem, this will also refresh self.df with new sample
        if update_self:
            self.use_coll(mongo_db=new_db, mongo_coll=new_coll)


    def make_cluster_df(self,
                        text_col: str = 'text_body',
                        id_col: str | None = 'bill_id',) -> pd.DataFrame:
        """ 
        Return the current collection with only two features: 
        > text_col: A field containing the document texts to be analyzed and clustered
        > id_col: [Optional] The key/id field (should be unique and non-null for all records)
        """
        _log(f"Creating DataFrame for clustering with columns: {f"{id_col}, {text_col}" if id_col else text_col}", self.verbose)
        
        if id_col:
            cols = [id_col, text_col]
        else:
            cols = [text_col]

        if self.local_mem:
            return(self.df[cols])
        else:
            return(self.get_records(pandas=True, inplace=False)[cols])


