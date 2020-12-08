# install api to get data from zillow
# pip install python-zillow

import zillow
import pprint

with open("./bin/config/zillow_key.conf", 'r') as f:
    key = f.readline().replace("\n", "")

api = zillow.ValuationApi()

address = "3400 Pacific Ave., Marina Del Rey, CA"
postal_code = "90292"

data = api.GetDeepSearchResults(key, address, postal_code)
pp = pprint.PrettyPrinter(indent=4)
pp.pprint(data.get_dict())

#
# deep_results = api.GetDeepSearchResults(key, "1920 1st Street South Apt 407, Minneapolis, MN", "55454")
# pp.pprint(deep_results.get_dict())