#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 15:27:04 2020

@author: jenn
"""
"""
# from __future__ import print_statement
import time
import swagger_client
from swagger_client.rest import ApiException
from pprint import pprint

# Configure OAuth2 access token for authorization: strava_oauth
swagger_client.configuration.access_token = '2453e1a6f852283b1ac37c10b8ac3301e95ff0a6'

# create an instance of the API class
api_instance = swagger_client.ClubsApi()
id = 494747 # Integer | The identifier of the club.
page = 1 # Integer | Page number. (optional)
perPage = 56 # Integer | Number of items per page. Defaults to 30. (optional) (default to 30)

try: 
    # List Club Activities
    api_response = api_instance.getClubActivitiesById(id, page=page, perPage=perPage)
    print(api_response)
except ApiException as e:
    print("Exception when calling ClubsApi->getClubActivitiesById: %s\n" % e)
""" 
    

import time
import swagger_client
from swagger_client.rest import ApiException
from pprint import pprint

# Configure OAuth2 access token for authorization: strava_oauth
configuration = swagger_client.Configuration()
configuration.access_token = '2453e1a6f852283b1ac37c10b8ac3301e95ff0a6'

# create an instance of the API class
api_instance = swagger_client.ClubsApi(swagger_client.ApiClient(configuration))
id = 494747 # int | The identifier of the club.
page = 1 # int | Page number. (optional)
per_page = 5 # int | Number of items per page. Defaults to 30. (optional) (default to 30)


try:
    # List Club Members
    api_response = api_instance.get_club_members_by_id(id, page=page, per_page=per_page)
    pprint(api_response)
except ApiException as e:
    print("Exception when calling ClubsApi->get_club_members_by_id: %s\n" % e)

"""
try:
    # List Club Activities
    api_response = api_instance.get_club_activities_by_id(id, page=page, per_page=per_page)
    pprint(api_response)
except ApiException as e:
    print("Exception when calling ClubsApi->get_club_activities_by_id: %s\n" % e)
"""


"""
# create an instance of the API class
api_instance = swagger_client.GearsApi(swagger_client.ApiClient(configuration))
id = 'b1231' # str | The identifier of the gear.

try:
    # Get Equipment
    api_response = api_instance.get_gear_by_id(id)
    pprint(api_response)
except ApiException as e:
    print("Exception when calling GearsApi->get_gear_by_id: %s\n" % e)
"""