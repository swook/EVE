"""Copyright 2020 ETH Zurich, Seonwook Park

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from collections import OrderedDict
import logging
import math
import os
import socket
import time
import traceback

import gspread
import numpy as np
from oauth2client.service_account import ServiceAccountCredentials

from core import DefaultConfig

config = DefaultConfig()
logger = logging.getLogger(__name__)


class GoogleSheetLogger:
    """Log selected outputs to a predefined Google Sheet.

    Many thanks to Emre for the idea and the initial code!
    """

    first_column_name = 'Identifier'

    def __init__(self, model):
        self.__model = model
        to_write = self.fill_in_basic_info()
        if config.resume_from == '':
            to_write['Start Time'] = time.strftime('%Y/%m/%d %H:%M:%S')

        # Write experiment information to create row for future logging
        try:
            self.ready = True
            self.update_or_append_row(to_write)
        except Exception:
            self.ready = False
            traceback.print_exc()
            return

    def fill_in_basic_info(self):
        to_write = OrderedDict()
        to_write['Identifier'] = self.__model.identifier
        to_write['Last Updated'] = time.strftime('%Y/%m/%d %H:%M:%S')

        # Write config parameters
        config_kv = config.get_all_key_values()
        config_kv = dict([
            (k, v) for k, v in config_kv.items()
            if not k.startswith('datasrc_')
            and not k.startswith('gsheet_')
        ])
        for k in sorted(list(config_kv.keys())):
            to_write[k] = config_kv[k]

        # Get hostname
        to_write['hostname'] = socket.getfqdn()

        # Get LSF job ID if exists
        if 'LSB_JOBID' in os.environ:
            to_write['LSF Job ID'] = os.environ['LSB_JOBID']

        return to_write

    def update_or_append_row(self, values):
        assert isinstance(values, dict)

        if not self.ready:  # Silently skip if init failed
            return

        if not os.path.isfile(config.gsheet_secrets_json_file):
            logger.info('Not logging to Google Sheets due to missing authentication information.')
            logger.info('> Please set the configuration entry: "gsheet_secrets_json_file".')
            return

        if len(config.gsheet_workbook_key) == 0:
            logger.info('Not logging to Google Sheets due to missing workbook key.')
            logger.info('> Please set the configuration entry: "gsheet_workbook_key".')
            return

        # Add some missing info automatically
        basic_info = self.fill_in_basic_info()
        for k, v in basic_info.items():
            values[k] = v

        # Authenticate
        try:
            credentials = ServiceAccountCredentials.from_json_keyfile_name(
                filename=config.gsheet_secrets_json_file,
                scopes=[
                    'https://www.googleapis.com/auth/spreadsheets',
                ],
            )
            client = gspread.authorize(credentials)
        except:  # noqa
            logger.debug('Could not authenticate with Drive API.')
            traceback.print_exc()
            return

        # Decide on sheet name to select
        sheet_name = self.__model.identifier.split('/')[0]

        # Find a workbook by name.
        workbook = client.open_by_key(config.gsheet_workbook_key)
        try:
            sheet = workbook.worksheet(sheet_name)
        except:  # noqa
            try:
                sheet = workbook.add_worksheet(title=sheet_name,
                                               rows=1000, cols=20)
            except:  # noqa
                logger.debug('Could not access/add worksheet.')
                traceback.print_exc()
                return
        try:
            current_values = sheet.get_all_values()
        except:  # noqa
            logger.debug('Could not get values from worksheet.')
            traceback.print_exc()
            return
        if len(current_values) == 0:
            try:
                sheet.update_cell(1, 1, self.first_column_name)
            except:  # noqa
                logger.debug('Could not insert first cell.')
                traceback.print_exc()
                return
            header = [self.first_column_name]
        else:
            header = current_values[0]

        identifier = values[self.first_column_name]

        # Construct new row
        is_header_changed = False
        new_row = [None] * len(header)
        for key, value in values.items():
            if key not in header:
                header.append(key)
                new_row.append(None)
                is_header_changed = True
            index = header.index(key)
            new_row[index] = value
            if isinstance(value, float) or isinstance(value, int):
                if math.isnan(value):
                    new_row[index] = 'NaN'
            elif isinstance(value, np.generic):
                if np.any(np.isnan(value)):
                    new_row[index] = 'NaN'
                elif np.isinf(value):
                    new_row[index] = 'Inf'
                else:
                    new_row[index] = np.asscalar(value)
            elif isinstance(value, np.ndarray) and value.ndim == 0:
                new_row[index] = value.item()
            elif hasattr(value, '__len__') and len(value) > 0:
                new_row[index] = str(value)

        # Update header as necessary
        cells_to_update = []
        if is_header_changed:
            cells_to_update += [
                gspread.models.Cell(1, col+1, value)
                for col, value in enumerate(header)
            ]

        # Either update an existing row or append new row
        try:
            row_index = [r[0] for r in current_values].index(identifier)
            cells_to_update += [
                gspread.models.Cell(row_index+1, col_index+1, value=value)
                for col_index, value in enumerate(new_row)
                if value is not None  # Don't remove existing values
            ]
        except:  # noqa
            sheet.append_row(new_row)

        # Run all necessary update operations
        if len(cells_to_update) > 0:
            try:
                sheet.update_cells(cells_to_update)
            except:  # noqa
                logger.debug('Error in API call to update cells.')
                traceback.print_exc()
                return
