import requests
import time

port = 24546
api_url = f'http://0.0.0.0:{port}/abort_request'
headers = {'content-type': 'application/json'}

payload = dict(
    # session_id = 1,
    abort_all=True,
)

while True:
    time.sleep(1)
    response = requests.post(api_url, headers=headers, json=payload)
    print(response)