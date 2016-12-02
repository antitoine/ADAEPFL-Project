import requests as rq
from bs4 import BeautifulSoup as bfs
import pandas as pd
import numpy as np
import time
import random
import os.path
import os
from Crypto.Cipher import AES
import binascii
import json
from datetime import datetime

KEY = '0187fcdf7ac3d90d68334afa03b87efe' # Decode Base64 string + Hex encoder
IV = '33a39433e4dde7c4ddb9d4502b8905d4' # Decode Base64 string + Hex encoder
KEY_BYTES = binascii.a2b_hex(KEY)
IV_BYTES = binascii.a2b_hex(IV)
CIPHER = AES.new(KEY_BYTES, AES.MODE_CBC, IV_BYTES, segment_size=128)

def unpad(string):
    '''
    Remove the PKCS#7 padding from a text string
    '''

    length = len(string)
    pad_size = string[-1]
    return string[:length - pad_size]


def decrypt_data(cipher, encrypted):
    '''
    Decrypt data given key and iv
    '''

    try:
        decrypted = cipher.decrypt(binascii.a2b_base64(encrypted).rstrip())
    except ValueError:
        return None

    return unpad(decrypted).decode()


def parse_html_runner_table(table, base_url='https://www.datasport.com'):

    n_columns = 0
    n_rows=0
    column_names = []

    # Find number of rows and columns
    # we also find the column titles if we can
    for row in table.find_all('tr'):

        # Determine the number of rows in the table
        td_tags = row.find_all('td')
        if len(td_tags) > 0:
            n_rows+=1
            if n_columns == 0:
                # Set the number of columns for our table
                n_columns = len(td_tags)

        # Handle column names if we find them
        th_tags = row.find_all('th') 
        if len(th_tags) > 0 and len(column_names) == 0:
            for th in th_tags:
                column_names.append(th.get_text())

    # Safeguard on Column Titles
    if len(column_names) > 0 and len(column_names) != n_columns:
        raise Exception("Column titles do not match the number of columns")

    columns = column_names if len(column_names) > 0 else range(0,n_columns)
    df = pd.DataFrame(columns = columns, index= range(0,n_rows))

    url_information_runner = []
    row_marker = 0
    for row in table.find_all('tr'):
        column_marker = 0

        columns = row.find_all('td')
        for index, column in enumerate(columns):
            df.iat[row_marker,column_marker] = column.get_text()

            # add the url to get informations about the runners.
            if(index == 1):
                url_information_runner.append(base_url + column.find('a')['href'])

            column_marker += 1
        if len(columns) > 0:
            row_marker += 1

    # Convert to float if possible
    for col in df:
        try:
            df[col] = df[col].astype(float)
        except ValueError:
            pass

    # add columns url
    df['url_run_event'] = pd.Series(url_information_runner)

    return df


def get_runners_information(acode_file, base_url='https://www.datasport.com/sys/myds/ajax/getDSInfo.htm?acode=', store_runners_information_file=False, store_runs_information_file=False):

    data_acode = pd.read_csv(acode_file)

    data_runners = []
    data_runs = []

    print('Start at: ' + str(datetime.now()), flush=True)
    print('Number of runners: ' + str(len(data_acode)), flush=True)
    
    for idx_acode, row_acode in data_acode.iterrows():
        
        url = base_url + row_acode['acode']
        
        print('[' + str(datetime.now()) + '] Runner acode ' + row_acode['acode'], end='', flush=True)
        
        # We retrieve filters and store cookies for further calls to server
        filters_page = rq.get(url)
        cookies = filters_page.cookies.get_dict()
        page = bfs(filters_page.text, 'html.parser')

        # We get the birthyear
        small_p = page.findAll('p', { 'class': 'small' })
        if len(small_p) >= 2:
            location = small_p[0].text.strip()
            birthyear = small_p[1].text.split(" ")[1:][0]
        else:
            print(' -:- No location of birthyear found', end='', flush=True)
            location = ''
            birthyear = ''

        # We store information of the runner
        data_runners.append({
                'acode': row_acode['acode'],
                'name': row_acode['name'],
                'birthyear': birthyear,
                'location': location
            })

        # We get information on the table
        table_run_event = page.find('table', {'id': 'timeTable'})

        try:
            information_runner = parse_html_runner_table(table_run_event)
        except Exception:
            print(' -:- Error when parse the html tablei: ' + url, flush=True)
            continue
        
        # We retrieve all runs of the runner
        print(' -:- Number of runs: ' + str(len(information_runner)), end='', flush=True)

        for idx_runner, row_runner in information_runner.iterrows():

            # Multiple try
            for i in range(0, 10):

                ajax_response = rq.get(row_runner['url_run_event'], cookies=cookies)

                # We generate a cipher to avoid any interference between the decrypt processes
                cipher = AES.new(KEY_BYTES, AES.MODE_CBC, IV_BYTES, segment_size=128)

                decrypted_response = decrypt_data(cipher=cipher, encrypted=ajax_response.text)
                time.sleep(random.uniform(0.5, 1))

                if decrypted_response != None:
                    break

            if decrypted_response == None:
                print(' -:- Error after multiple try to decrypt from: ' + row_runner['url_run_event'] + ' / Runner: ' + url + ' / Encrypted response: "' + ajax_response.text + '"', end='', flush=True)
                continue

            try:
                running_information = json.loads(decrypted_response)
            except Exception:
                print(' -:- Error when handling json response from: ' + row_runner['url_run_event'] + ' / Runner: ' + url + ' / Encrypted response: "' + ajax_response.text + '"', end='', flush=True)
                continue
            
            data_runs.append({**running_information, 'acode': row_acode['acode']})
        
        time.sleep(random.uniform(0.5, 1))
        print(' -:- End at: ' + str(datetime.now()), flush=True)

    df_data_runners = pd.DataFrame(data_runners)
    df_data_runs = pd.DataFrame(data_runs)
    
    if store_runners_information_file:
        df_data_runners.to_csv(store_runners_information_file, index=False)
    if store_runs_information_file:
        df_data_runs.to_csv(store_runs_information_file, index=False)

    return [df_data_runners, df_data_runs]


df_data_runners, df_data_runs = get_runners_information(acode_file='./acode.csv', store_runners_information_file='./runners.csv', store_runs_information_file='./runs.csv')

print('End at: ' + str(datetime.now()), flush=True)
