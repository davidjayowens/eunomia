"""
Part of:
EUNOMIA AI
A project to identify who writes our laws.

Created Feb 2022

@author: David Owens

Significant credit goes to Chris Poliquin (Git: @poliquin) for their pylegiscan package.
https://github.com/poliquin/pylegiscan

NOTE: The 
"""


#from os import listdir, environ
#from os.path import isfile, join
from pathlib import Path
import json
#import pymongo
#from urllib.parse import urlencode
#import requests
import re   # regex search
#from base64 import b64decode    # decode bill text
#from bs4 import BeautifulSoup as BS     # strip HTML in bill text

from djo.api import get
""" Personal method for API GET requests
get(url: str, 
    headers: dict | None = None,
    results_as: Literal['bytes', 'dict', 'raw', 'str'] = 'dict',
    timeout: int | float = 10,
    stream: bool = False,
    **params)
"""

"""
Example record from BillPrep:
        bill_data = {
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
                    "text_body": None, # TODO: "text_body" will be created
                }
"""


__author__  = 'David Owens'
__date__    = '2025.12.24'
__version__ = __date__


# List of state abbreviations (includes DC, no Puerto Rico)
STATES = ['ak', 'al', 'ar', 'az', 'ca', 'co', 'ct', 'dc', 'de', 'fl', 'ga',
          'hi', 'ia', 'id', 'il', 'in', 'ks', 'ky', 'la', 'ma', 'md', 'me',
          'mi', 'mn', 'mo', 'ms', 'mt', 'nc', 'nd', 'ne', 'nh', 'nj', 'nm',
          'nv', 'ny', 'oh', 'ok', 'or', 'pa', 'ri', 'sc', 'sd', 'tn', 'tx',
          'ut', 'va', 'vt', 'wa', 'wi', 'wv', 'wy']

# Legiscan state ID mapping
STATE_IDS = {2:'ak', 1:'al', 4:'ar', 3:'az', 5:'ca', 6:'co', 7:'ct', 51:'dc', 8:'de', 9:'fl', 10:'ga',
        11:'hi', 15:'ia', 12:'id', 13:'il', 14:'in', 16:'ks', 17:'ky', 18:'la', 21:'ma', 20:'md', 19:'me',
        22:'mi', 23:'mn', 25:'mo', 24:'ms', 26:'mt', 33:'nc', 34:'nd', 27:'ne', 29:'nh', 30:'nj', 31:'nm',
        28:'nv', 32:'ny', 35:'oh', 36:'ok', 37:'or', 38:'pa', 39:'ri', 40:'sc', 41:'sd', 42:'tn', 43:'tx',
        44:'ut', 46:'va', 45:'vt', 47:'wa', 49:'wi', 48:'wv', 50:'wy'}

# Legiscan bill_status mapping (defines current aggregate status of a bill)
BILL_STATUS = {1: "Introduced",
               2: "Engrossed",
               3: "Enrolled",
               4: "Passed",
               5: "Vetoed",
               6: "Failed/Dead"}

# Legiscan bill_progress mapping (defines steps in the legislative process)
BILL_PROGRESS = {1: "Introduced",
                 2: "Engrossed",
                 3: "Enrolled",
                 4: "Passed",
                 5: "Vetoed",
                 6: "Failed/Dead",
                 7: "Veto Override",
                 8: "Chapter/Act/Statute",
                 9: "Committee Referral",
                10: "Committee Report Pass",
                11: "Committee Report DNP"}

# Legiscan bill_type mapping
BILL_TYPES =   { 1: 'Bill',
                 2: 'Resolution',
                 3: 'Concurrent Resolution',
                 4: 'Joint Resolution',
                 5: 'Joint Resolution Constitutional Amendment',
                 6: 'Executive Order',
                 7: 'Constitutional Amendment',
                 8: 'Memorial',
                 9: 'Claim',
                10: 'Commendation',
                11: 'Committee Study Request',
                12: 'Joint Memorial',
                13: 'Proclamation',
                14: 'Study Request',
                15: 'Address',
                16: 'Concurrent Memorial',
                17: 'Initiative',
                18: 'Petition',
                19: 'Study Bill',
                20: 'Initiative Petition',
                21: 'Repeal Bill',
                22: 'Remonstration',
                23: 'Committee Bill'}

class LegiScanError(Exception):
    pass

class Legiscanner:
    URL = 'http://api.legiscan.com'

    def __init__(self, 
                 api_key: str) -> object:
        """ Initialize the Legiscanner object using an API key.
        """
        self._key = api_key.strip()


    def get_sessions(self, 
                     state: str | None = None) -> list:
        """ 
        Get session_id(s) via Legiscan's getSessionList API.
        Also includes session years, titles, and "special" flags.

        Parameters
        ----------
        state : str, optional
            State to get sessions for; must be two-letter state abbreviation.

        Returns
        -------
        eg: [
            {   'session_id': 1476,
                'state_id': 36,
                'state_abbr': 'OK',
                'year_start': 2017,
                'year_end': 2017,
                'prefile': 0,
                'sine_die': 1,
                'prior': 1,
                'special': 1,
                'session_tag': '1st Special Session',
                'session_title': '2017 1st Special Session',
                'session_name': '2017 First Special Session',
                'dataset_hash': 'af84a3575cce11ebc3899c90df1cd4be',
                'session_hash': 'af84a3575cce11ebc3899c90df1cd4be',
                'name': '2017 Special Session'
            },
            {   'session_id': 1260,
                'state_id': 36,
                'state_abbr': 'OK',
                'year_start': 2017,
                'year_end': 2017,
                'prefile': 0,
                'sine_die': 1,
                'prior': 1,
                'special': 0,
                'session_tag': 'Regular Session',
                'session_title': '2017 Regular Session',
                'session_name': '2017 Regular Session',
                'dataset_hash': '19c654648a02d1d0f0dac7bdd45d339f',
                'session_hash': '19c654648a02d1d0f0dac7bdd45d339f',
                'name': '2017 Regular Session'
            },
            ...
        ]
        """
        if isinstance(state, str):
            state = state.strip()[:2].lower()

        data = get(url=self.URL, key=self._key, op='getSessionList', state=state)

        return(data['sessions'])


    def get_session_details(self, 
                            session_id: int | str) -> dict:
        """ 
        Use session_id to get metadata for bills in the target session.
        Uses Legiscan's getMasterList API endpoint.
        

        Parameters
        ----------
        session_id : int or str
            The session_id to query.

        Returns
        -------
        eg: {
            'session': {
                'session_id': 2218,
                'state_id': 1,
                'year_start': 2026,
                'year_end': 2026,
                'prefile': 1,
                'sine_die': 0,
                'prior': 0,
                'special': 0,
                'session_tag': 'Regular Session',
                'session_title': '2026 Regular Session',
                'session_name': '2026 Regular Session'
            },
            '0': {
                'bill_id': 2038126,
                'number': 'HB1',
                'change_hash': 'd417aeef588adda4e2e240b597ae91e4',
                'url': 'https://legiscan.com/AL/bill/HB1/2026',
                'status_date': '2026-01-13',
                'status': 1,
                'last_action_date': '2026-01-13',
                'last_action': 'Pending House Public Safety and Homeland Security',
                'title': 'Motor vehicles; suspension of driver license and ignition interlock device following first driving while under the influence conviction required',
                'description': 'Motor vehicles; suspension of driver license and ignition interlock device following first driving while under the influence conviction required'
            },
            '1': {
                'bill_id': 2038144,
                'number': 'HB2',
                'change_hash': '93acfe934fb9eaff5ceb59b6cd2b1c70',
                'url': 'https://legiscan.com/AL/bill/HB2/2026',
                'status_date': '2026-01-13',
                'status': 1,
                'last_action_date': '2026-01-13',
                'last_action': 'Pending House State Government',
                'title': 'Gulf of Mexico, renamed, observation and implementation by state and local entities and state and local employees required where practicable',
                'description': 'Gulf of Mexico, renamed, observation and implementation by state and local entities and state and local employees required where practicable'
            },
            ...
        }
        """
        data = get(url=self.URL, key=self._key, op='getMasterList', id=session_id)

        return(data['masterlist'])


    def get_bill_details(self, 
                         bill_id: int | str) -> dict:
        """ Uses bill_id to get doc_id(s) via getBill API. Also
            includes metatext details of the bill.
            Eg: get_bill_details(123456789)

        Returns
        -------
        eg: {
            'bill_id': 2038144,
            'change_hash': '93acfe934fb9eaff5ceb59b6cd2b1c70',
            'session_id': 2218,
            'session': {
                'session_id': 2218,
                'state_id': 1,
                'year_start': 2026,
                'year_end': 2026,
                'prefile': 1,
                'sine_die': 0,
                'prior': 0,
                'special': 0,
                'session_tag': 'Regular Session',
                'session_title': '2026 Regular Session',
                'session_name': '2026 Regular Session'},
            'url': 'https://legiscan.com/AL/bill/HB2/2026',
            'state_link': 'https://alison.legislature.state.al.us/bill-search',
            'completed': 0,
            'status': 1,
            'status_date': '2026-01-13',
            'progress': [
                {'date': '2026-01-13', 'event': 1},
                {'date': '2026-01-13', 'event': 9}
            ],
            'state': 'AL',
            'state_id': 1,
            'bill_number': 'HB2',
            'bill_type': 'B',
            'bill_type_id': '1',
            'body': 'H',
            'body_id': 11,
            'current_body': 'H',
            'current_body_id': 11,
            'title': 'Gulf of Mexico, renamed, observation and implementation by state and local entities and state and local employees required where practicable',
            'description': 'Gulf of Mexico, renamed, observation and implementation by state and local entities and state and local employees required where practicable',
            'pending_committee_id': 2537,
            'committee': {
                'committee_id': 2537,
                'chamber': 'H',
                'chamber_id': 11,
                'name': 'State Government'},
            'referrals': [
                {   'date': '2026-01-13',
                    'committee_id': 2537,
                    'chamber': 'H',
                    'chamber_id': 11,
                    'name': 'State Government'}
            ],
            'history': [
                {   'date': '2025-06-25',
                    'action': 'Prefiled',
                    'chamber': 'H',
                    'chamber_id': 11,
                    'importance': 0},
                {   'date': '2026-01-13',
                    'action': 'Read for the first time and referred to the House Committee on State Government',
                    'chamber': 'H',
                    'chamber_id': 11,
                    'importance': 1},
                ...
            ],
            'sponsors': [
                {'people_id': 15745,
                'person_hash': 'xzdz5vqm',
                'party_id': '2',
                'state_id': 1,
                'party': 'R',
                'role_id': 1,
                'role': 'Rep',
                'name': 'David Standridge',
                'first_name': 'David',
                'middle_name': '',
                'last_name': 'Standridge',
                'suffix': '',
                'nickname': '',
                'district': 'HD-034',
                'ftm_eid': 15881725,
                'votesmart_id': 82406,
                'opensecrets_id': '',
                'knowwho_pid': 446076,
                'ballotpedia': 'David_Standridge',
                'bioguide_id': '',
                'sponsor_type_id': 1,
                'sponsor_order': 1,
                'committee_sponsor': 0,
                'committee_id': 0,
                'state_federal': 0,
                'bio': 
                    {'social': 
                        {'capitol_phone': '334-261-0446',
                        'district_phone': '205-543-0647',
                        'email': 'david.standridge@alhouse.gov',
                        'webmail': '',
                        'biography': 'https://alison.legislature.state.al.us/house-leaders-members?tab=1',
                        'image': 'https://alison.legislature.state.al.us/files/pdf/house/members/Standridge_34.png',
                        'ballotpedia': 'https://ballotpedia.org/David_Standridge',
                        'votesmart': 'https://justfacts.votesmart.org/candidate/biography/82406'},
                        'capitol_address': {'address1': '11 South Union Street',
                        'address2': 'Suite 403-B',
                        'city': 'Montgomery',
                        'state': 'AL',
                        'zip': '36130'},
                    'links': 
                        {'official': 
                            {'bluesky': '',
                            'facebook': 'https://www.facebook.com/JudgeStandridge',
                            'instagram': 'https://www.instagram.com/judgestandridge/',
                            'linkedin': '',
                            'tiktok': '',
                            'twitter': 'https://www.twitter.com/JudgeStandridge',
                            'website': '',
                            'youtube': ''},
                        'personal': 
                            {'bluesky': '',
                            'facebook': 'https://www.facebook.com/david.standridge.35',
                            'instagram': '',
                            'linkedin': '',
                            'tiktok': '',
                            'twitter': '',
                            'website': '',
                            'youtube': ''}
                        }
                    }
                }
            ],
            'sasts': [],
            'subjects': [
                {'subject_id': 375031,
                'subject_name': 'Government Administration'}
            ],
            'texts': [
                {'doc_id': 3258583,
                'date': '2025-06-25',
                'type': 'Introduced',
                'type_id': 1,
                'mime': 'application/pdf',
                'mime_id': 2,
                'url': 'https://legiscan.com/AL/text/HB2/id/3258583',
                'state_link': 'https://alison.legislature.state.al.us/files/pdf/SearchableInstruments/2026RS/HB2-int.pdf',
                'text_size': 501505,
                'text_hash': '3b17ec5825307af6c7ef1f0610497fa6',
                'alt_bill_text': 0,
                'alt_mime': '',
                'alt_mime_id': 0,
                'alt_state_link': '',
                'alt_text_size': 0,
                'alt_text_hash': ''}
            ],
            'votes': [],
            'amendments': [],
            'supplements': [],
            'calendar': []
        }
        """
        data = get(url=self.URL, key=self._key, op='getBill', id=bill_id)

        return(data['bill'])


    def get_bill_text(self, 
                      doc_id: int | str) -> dict:
        """ Uses doc_id to get bill text via getBillText API.
            Eg: get_bill_text(123456789)

        Parameters
        ----------
        doc_id : int or str
            The doc_id to query.

        Returns
        -------
        eg: {
            "doc_id": 647508,
            "bill_id": 502329,
            "date": "2012-05-23",
            "type": "Enrolled",
            "type_id": "5,
            "mime": "application/pdf",
            "mime_id": 2,
            "text_size": 1724832,
            "text_hash": "bcd4ee6b03d1ae74a8b225f9b85b6863",
            "doc": "MIME 64 Encoded Document”
        }
        """
        data = get(url=self.URL, key=self._key, op='getBillText', id=doc_id)

        return(data['text'])
    

    def get_amendment(self,
                      am_id: int | str) -> dict:
        """
        Docstring for get_bill_amendment
        
        Parameters
        ----------
        am_id : int or str
            The amendment_id to query.

        Returns
        -------
        eg: {
            "amendment_id": 37508,
            "chamber": "S",
            "chamber_id": 36,
            "bill_id": 852200,
            "adopted": 0,
            "date": "2016-04-01",
            "title": "Senate Amendment 001",
            "description": "Senate Amendment 001",
            "mime": "text/html",
            "mime_id": 1,
            "amendment_size": 6914,
            "amendment_hash": "350a599b8db27a1999d17efd71b420d6",
            "doc": "MIME 64 Encoded Document”
        }
        """
        data = get(url=self.URL, key=self._key, op='getAmendment', id=am_id)

        return(data['amendment'])


    def get_supplement(self,
                       supp_id: int | str) -> dict:
        """
        Get supplemental documents, such as fiscal notes, veto letters, etc.

        Parameters
        ----------
        supp_id : int or str
            The supplement_id to query.

        Returns
        -------
        eg: {
            "supplement_id": 47508,
            "bill_id": 853693,
            "date": "0000-00-00",
            "type_id": 3,
            "type": "Fiscal Note/Analysis",
            "title": "Analysis",
            "description": "Fiscal Note/Analysis",
            "mime": "application/pdf",
            "mime_id": 2,
            "supplement_size": 134305,
            "supplement_hash": "ba83212a92b26b34d3a1b751e0d45ad1",
            "doc": "MIME 64 Encoded Document”
        }
        """
        data = get(url=self.URL, key=self._key, op='getSupplement', id=supp_id)

        return(data['supplement'])


    def get_votes(self,
                  call_id: int | str) -> dict:
        """
        Get supplemental documents, such as fiscal notes, veto letters, etc.

        Parameters
        ----------
        call_id : int or str
            The roll_call_id to query.

        Returns
        -------
        eg: {
            "roll_call_id": 234223,
            "bill_id": 460445,
            "date": "2013-02-20",
            "desc": "House: Human Services Subcommittee: DO PASS",
            "yea": 2,
            "nay": 1,
            "nv": 1,
            "absent": 1,
            "total": 5,
            "passed": 1,
            "chamber": "H",
            "chamber_id": 79,
            "votes":[
                {   "people_id": 3709,
                    "vote_id": 1,
                    "vote_text": "Yea"
                },
                {   "people_id": 3715,
                    "vote_id": 2,
                    "vote_text": "Nay"
                },
                ...
            ]
        }
        """
        data = get(url=self.URL, key=self._key, op='getRollCall', id=call_id)

        return(data['roll_call'])


    def get_person(self,
                   ppl_id: int | str) -> dict:
        """
        Get supplemental documents, such as fiscal notes, veto letters, etc.

        Parameters
        ----------
        ppl_id : int or str
            The people_id to query.

        Returns
        -------
        eg: {
            "people_id": 16788,
            "person_hash": " 1s25cljm",
            "state_id": 46,
            "party_id": "1",
            "party": "D",
            "role_id": 1,
            "role": "Rep",
            "name": " Joseph Preston",
            "first_name": " Joseph ",
            "middle_name": "E.",
            "last_name": "Preston",
            "suffix": "",
            "nickname": "",
            "district": "HD-063",
            "ftm_eid": 7290094,
            "votesmart_id": 154843,
            "opensecrets_id": "",
            "knowwho_pid": 523841,
            "ballotpedia": " Joseph_Preston_(Virginia)",
            "committee_sponsor": 0,
            "committee_id": 0
        }
        """
        data = get(url=self.URL, key=self._key, op='getPerson', id=ppl_id)

        return(data['person'])


    def get_dataset_list(self,
                         state: str | None = None,
                         year: int | str | None = None) -> list:
        """
        Get list of available datasets for bulk download.

        Parameters
        ----------
        state : str, optional
            Filter results to the given state; must be a 2-letter abbrevation.

        year : int or str, optional
            Filter results to the given year; must be a 4-digit number.

        Returns
        -------
        eg: [
            {   "state_id": 5,
                "session_id": 1624,
                "special": 0,
                "year_start": 2019,
                "year_end": 2020,
                "session_name": "2019-2020 Regular Session",
                "session_title": "Regular Session",
                "dataset_hash": "e0f2b493b637ec870ca886931bfa9896",
                "dataset_date": "2020-01-19",
                "dataset_size": 11958086,
                "access_key": "3Qd0kRszXtZuRloonDQx63",
            },
            {   "state_id": 5,
                "session_id": 1400,
                "special": 0,
                "year_start": 2017,
                "year_end": 2018,
                "session_name": "2017-2018 Regular Session",
                "session_title": "Regular Session",
                "dataset_hash": "64eaa5580417096fdc2b3d06402b2841",
                "dataset_date": "2018-12-09",
                "dataset_size": 22138644,
                "access_key": "5R6lftn5PdMvoZF9yMsE9V",
            },
            ...
        ]
        """
        data = get(url=self.URL, key=self._key, op='getDatasetList', state=state, year=year)

        return(data['datasetlist'])
    

    def get_dataset(self,
                    session_id: int | str,
                    access_key: str) -> dict:
        """
        Returns a base64 encoded zip binary containing the target dataset.

        Parameters
        ----------
        session_id : int or str
            Retrieve dataset for given session_id.

        access_key : str
            Access key from get_dataset_list() for the session_id being requested.

        Returns
        -------
        eg: {
            "state_id": 5,
            "session_id": 1624,
            "session_name": "2019-2020 Regular Session",
            "dataset_hash": "1c7d77fe298a4d30ad763733ab2f8c84",
            "dataset_date": "2018-12-23",
            "dataset_size": 317775,
            "mime": "application/zip",
            "zip": "MIME 64 Encoded ZIP Archive"
        }
        """
        data = get(url=self.URL, key=self._key, op='getDataset', id=session_id, access_key=access_key)

        return(data['dataset'])
