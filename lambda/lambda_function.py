"""
Copyright 2024 Amazon.com, Inc. or its affiliates.  All Rights Reserved.
SPDX-License-Identifier: MIT-0
"""
import boto3
import json
import logging
import os
import requests


# Set the logger
def set_log_config(logger_obj):
    log_level = os.environ['LOG_LEVEL']
    if log_level.upper() == 'NOTSET':
        logger_obj.setLevel(logging.NOTSET)
    elif log_level.upper() == 'DEBUG':
        logger_obj.setLevel(logging.DEBUG)
    elif log_level.upper() == 'INFO':
        logger_obj.setLevel(logging.INFO)
    elif log_level.upper() == 'WARNING':
        logger_obj.setLevel(logging.WARNING)
    elif log_level.upper() == 'ERROR':
        logger_obj.setLevel(logging.ERROR)
    elif log_level.upper() == 'CRITICAL':
        logger_obj.setLevel(logging.CRITICAL)
    else:
        logger_obj.setLevel(logging.NOTSET)


# Initialize the logger
logger = logging.getLogger()
set_log_config(logger)


# Perform Wikipedia search
def perform_wikipedia_search(search_string):
    max_results = int(os.environ['MAX_NUMBER_OF_RESULTS'])
    logging.info('Performing search on Wikipedia with search string "{}" and max results as "{}"...'.
                 format(search_string, max_results))
    excerpts = ''
    headers = {
        'User-Agent': os.environ['APP_NAME']
    }
    parameters = {'q': search_string, 'limit': max_results}
    response = requests.get(os.environ['WIKIPEDIA_ENDPOINT_URL'], headers=headers, params=parameters)
    response = json.loads(response.text)
    for page in response['pages']:
        if 'excerpt' in page:
            excerpts += page['excerpt'] + '\n\n'
    results = {"context": excerpts}
    logging.info('Completed performing search on Wikipedia.')
    return results


# Parse the input Lambda event received from Agents for Amazon Bedrock
def parse_request_and_prepare_response(event):
    logging.info('Parsing request data and preparing response...')
    search_string = ""
    # Loop through the input parameters
    input_parameters = event["parameters"]
    for input_parameter in input_parameters:
        # Retrieve the value of the search string
        if input_parameter["name"] == "SearchString":
            search_string = input_parameter["value"]
            break
    response_body = {
        'TEXT': {
            'body': json.dumps(perform_wikipedia_search(search_string))
        }
    }
    response = {
        "actionGroup": event["actionGroup"],
        "function": event["function"],
        "functionResponse": {
            "responseBody": response_body
        }
    }
    session_attributes = event["sessionAttributes"]
    prompt_session_attributes = event["promptSessionAttributes"]
    logging.info('Completed parsing request data and preparing response.')
    return {
        "messageVersion": "1.0",
        "response": response,
        "sessionAttributes": session_attributes,
        "promptSessionAttributes": prompt_session_attributes,
    }


# The handler function
def lambda_handler(event,context):
    logging.info('Executing the handler() function...')
    logging.info('Request event :: {}'.format(event))
    logging.info('Request context :: {}'.format(context))
    # Parse the request data and prepare response
    return_data = parse_request_and_prepare_response(event)
    logging.info('Response :: {}'.format(return_data))
    logging.info('Completed executing the handler() function.')
    return return_data
