from __future__ import print_function

import os.path
import pickle

import pandas as pd
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# If modifying these scopes, delete the file token.pickle.
SCOPES = ['https://www.googleapis.com/auth/spreadsheets']

# The ID and range of a sample spreadsheet.
# SAMPLE_SPREADSHEET_ID = '1TpceCGWnEqansqw0rrLAHz-xUWEVj7Qgri9Zic6osws'
SAMPLE_RANGE_NAME = 'Sheet1!A2:K'
value_input_option = 'RAW'


def get_row_number(row):
    return int(row.name) + 2


class GoogleSheetService:
    service = None
    spreadsheet_id = ''

    def __init__(self, spreadsheet_id) -> None:
        creds = None
        # The file token.pickle stores the user's access and refresh tokens, and is
        # created automatically when the authorization flow completes for the first
        # time.
        if os.path.exists('token.pickle'):
            with open('token.pickle', 'rb') as token:
                creds = pickle.load(token)
        # If there are no (valid) credentials available, let the user log in.
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    'credentials.json', SCOPES)
                creds = flow.run_console()
            # Save the credentials for the next run
            with open('token.pickle', 'wb') as token:
                pickle.dump(creds, token)

        self.service = build('sheets', 'v4', credentials=creds)
        self.spreadsheet_id = spreadsheet_id

    def get_google_sheet(self):
        """Shows basic usage of the Sheets API.
        Prints values from a sample spreadsheet.
        """

        # Call the Sheets API
        sheet = self.service.spreadsheets()
        result = sheet.values().get(spreadsheetId=self.spreadsheet_id,
                                    range=SAMPLE_RANGE_NAME).execute()
        values = result.get('values', [])

        if not values:
            print('No data found.')
            return None
        else:
            # print('Experiment,\tPolicy,\tData,\tzeta_nb_steps,\teps,\tenv_suffix,\tenvname,\ttmux,\tstatus,\tcommand')
            # for row in values:
            # Print columns A and E, which correspond to indices 0 and 4.
            # print('%s,\t%s,\t%s,\t%s,\t%s,\t%s,\t%s,\t%s,\t%s,\t%s' % (
            #     row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9]))
            df = self.convert_to_df(values)
            return df

    def convert_to_df(self, values: list):
        d = pd.DataFrame(values)
        d.columns = ['Experiment', 'Policy', 'Data', 'zeta_nb_steps', 'eps', 'env_suffix', 'envname', 'tmux', 'status',
                     'priority', 'command']

        return d

    def get_next_pending(self, df: pd.DataFrame = None):
        if df is None:
            df = self.get_google_sheet()

        pendings = df[df['status'] == 'pending']

        sorted_pendings = pendings.sort_values(by=['priority', 'Experiment'], ascending=[False, True])

        if len(sorted_pendings) > 0:
            r = sorted_pendings.iloc[0]
            return r
        else:
            return None

    def update_status(self, status: str, row):
        COLUMN = 'I'
        self.update_cell(row, COLUMN, status)

    def update_tmux(self, tmux: str, row):
        COLUMN = 'H'
        self.update_cell(row, COLUMN, tmux)

    def update_cell(self, row, column, value):
        values = [
            [
                value
            ],
            # Additional rows ...
        ]
        body = {
            'values': values
        }
        result = self.service.spreadsheets().values().update(
            spreadsheetId=self.spreadsheet_id, range="Sheet1!{}{}:{}".format(column, row, column),
            valueInputOption=value_input_option, body=body).execute()
