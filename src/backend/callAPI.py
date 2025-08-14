import requests
import warnings
import os
warnings.filterwarnings("ignore")
HOST_NAME = os.environ['HOST_NAME'] if 'HOST_NAME' in os.environ else '45.129.46.3'

api_url = "http://" + HOST_NAME + "/invocations"

#request = {"user_input": "xgboost"}
sent = "xgboost"
request = { "user_input": [sent]}


response = requests.post(api_url, json=request)
print("\n")
print("######################################### Output ###########################################")
print("\n")
# print("Response Code = ", response)
dict_ = response.json()

# articles = dict_['Topic'].items()
print(dict_)