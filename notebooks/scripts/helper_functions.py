"""
Copyright 2024 Amazon.com, Inc. or its affiliates.  All Rights Reserved.
SPDX-License-Identifier: MIT-0
"""
import json
from langchain.prompts.prompt import PromptTemplate
from langchain_community.llms import Bedrock
from langchain_aws import ChatBedrock
from langchain_aws import AmazonKnowledgeBasesRetriever
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage
import logging
import os
import requests
import sagemaker
import time
import uuid
from statistics import mean, median, mode
from timeit import default_timer as timer

# Create the logger
DEFAULT_LOG_LEVEL = logging.NOTSET
DEFAULT_LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
log_level = os.environ.get('LOG_LEVEL')
match log_level:
    case '10':
        log_level = logging.DEBUG
    case '20':
        log_level = logging.INFO
    case '30':
        log_level = logging.WARNING
    case '40':
        log_level = logging.ERROR
    case '50':
        log_level = logging.CRITICAL
    case _:
        log_level = DEFAULT_LOG_LEVEL
log_format = os.environ.get('LOG_FORMAT')
if log_format is None:
    log_format = DEFAULT_LOG_FORMAT
elif len(log_format) == 0:
    log_format = DEFAULT_LOG_FORMAT
# Set the basic config for the logger
logging.basicConfig(level=log_level, format=log_format)


def download_file(download_url, dir_name):
    """
    Function to download the specified file to the specified directory

    Parameters:
    download_url (string): The URL of the file to download
    dir_name (string): The name of the directory where the file should be downloaded

    Returns:
    string: The name of the downloaded file
    """
    # Download the file
    response_content = requests.get(download_url).content
    # Write to local directory
    file_name = download_url.split("/")[-1]
    with open('{}/{}'.format(dir_name, file_name), "wb") as f:
        f.write(response_content)
        logging.info("Downloaded file '{}' to '{}'.".format(file_name, dir_name))
    return file_name


def upload_to_s3(dir_name, s3_bucket_name, s3_key_prefix):
    """
    Function to upload the specified file to the specified S3 location

    Parameters:
    dir_name (string): The name of the directory from where the file should be uploaded
    s3_bucket_name (string): The S3 bucket name
    s3_key_prefix (string): The name of the file which will also serve as the S3 key prefix

    Returns:
    None
    """
    # Upload to S3
    data_s3_path = sagemaker.Session().upload_data(path='{}/'.format(dir_name),
                                                   bucket=s3_bucket_name,
                                                   key_prefix=s3_key_prefix)
    logging.info("Uploaded file(s) from '{}' to '{}'.".format(dir_name, data_s3_path))


def delete_s3_object(s3_client, s3_bucket_name, object_key):
    """
    Function to delete the specified object to the specified S3 location

    Parameters:
    s3_client (boto3 client): The boto3 client for S3
    s3_bucket_name (string): The S3 bucket name
    object_key (string): The key to the S3 object

    Returns:
    None
    """
    # Delete S3 object
    s3_client.delete_object(
        Bucket=s3_bucket_name,
        Key=object_key,
    )
    logging.info("Deleted object '{}' from S3 bucket '{}'.".format(object_key, s3_bucket_name))


def delete_aoss_collection(aoss_client, collection_id):
    """
    Function to delete the specified Amazon OpenSearch Serverless (AOSS) collection

    Parameters:
    aoss_client (boto3 client): The boto3 client for Amazon OpenSearch Serverless (AOSS)
    collection_id (string): The id of the AOSS collection

    Returns:
    None
    """
    # Delete AOSS collection
    aoss_client.delete_collection(
        id=collection_id
    )
    logging.info("Initiated deletion of AOSS collection '{}'.".format(collection_id))


def print_results_relevancy_score_stats(query, query_results, threshold_score):
    """
    Function to print the statistics of the query results relevancy scores

    Parameters:
    query (string): The query that was sent by the user
    query_results (list): The list of query result dict objects
    threshold_score (float): The threshold value for query to query result relevancy

    Returns:
    None
    """
    if len(query_results) > 0:
        scores = []
        threshold_score_results_count = 0
        for query_result in query_results:
            scores.append(query_result['relevancy_score'])
            if query_result['relevancy_score'] >= threshold_score:
                threshold_score_results_count += 1
        logging.info(("\nQuery: {}\nTotal number of query results: {}" +
                      "\nThreshold relevancy score: {}\nTotal number of query results meeting or exceeding threshold: {}" +
                      "\n#### Relevancy score statistics ####\nRelevancy scores: {}\nMean: {}\nMedian: {}\nMode: {}\n").
                     format(query, len(scores), threshold_score, threshold_score_results_count, scores, mean(scores),
                            median(scores), mode(scores)))
    else:
        logging.warning("No query result found.")


def filter_query_results_by_threshold(query_results, threshold_score):
    """
    Function to filter the query results whose relevancy score meets or exceeds the specified threshold

    Parameters:
    query_results (list): The list of query result dict objects
    threshold_score (float): The threshold value for query to query result relevancy

    Returns:
    list: The list of query result dict objects where the relevancy score is at or above the threshold value
    """
    filtered_query_results = []
    for query_result in query_results:
        if query_result['relevancy_score'] >= threshold_score:
            filtered_query_results.append(query_result)
    return filtered_query_results


def get_s3_bucket_name_from_arn(s3_arn):
    """
    Function to prepare the prompt

    Parameters:
    s3_arn (string): The ARN of the S3 bucket

    Returns:
    string: The S3 bucket name
    """
    return (s3_arn.split('arn:aws:s3:::', 1))[1]


def print_claude_3_llm_info(bedrock_client, inf_type):
    """
    Function to print the specified type of Anthropic Claude 3 models

    Parameters:
    bedrock_client (boto3 client): The boto3 client for Bedrock
    inf_type (string): The inference type - on-demand or provisioned

    Returns:
    None
    """
    # List all the available Anthropic Claude 3 LLMs in Amazon Bedrock with
    # the specified throughput (inference type) pricing
    models_info = ''
    response = bedrock_client.list_foundation_models(byProvider="Anthropic",
                                                     byInferenceType=inf_type)
    model_summaries = response["modelSummaries"]
    models_info = models_info + "\n"
    models_info = models_info + "-".ljust(125, "-") + "\n"
    models_info = models_info + "{:<15} {:<30} {:<20} {:<20} {:<40}".format("Provider Name", "Model Name",
                                                                            "Input Modalities",
                                                                            "Output Modalities", "Model Id")
    models_info = models_info + "-".ljust(125, "-")
    for model_summary in model_summaries:
        # Check for Claude 3 LLMs and process
        if model_summary["modelName"] in ("Claude 3 Sonnet", "Claude 3 Haiku"):
            models_info = models_info + "\n"
            models_info = models_info + "{:<15} {:<30} {:<20} {:<20} {:<40}".format(model_summary["providerName"],
                                                                                    model_summary["modelName"],
                                                                                    "|".join(model_summary[
                                                                                                 "inputModalities"]),
                                                                                    "|".join(model_summary[
                                                                                                 "outputModalities"]),
                                                                                    model_summary["modelId"])
    models_info = models_info + "-".ljust(125, "-") + "\n"
    logging.info("Displaying available Anthropic Claude 3 models in the '{}' Region:".
                 format(bedrock_client.meta.region_name) + models_info)


def get_kb_that_meets_requirements(bedrock_agt_client, kb_id):
    """
    Function to get the KB that meets the requirements

    Parameters:
    bedrock_agt_client (boto3 client): The boto3 client for Bedrock Agent
    kb_id (string): The id of the Knowledge Base

    Returns:
    string: The id of the Knowledge Base
    string: The id of the data source associated with the above Knowledge Base
    string: The name of the S3 bucket that is associated with the above data source
    string: The ARN of the Amazon OpenSearch Serverless (AOSS) collection associated with the above Knowledge Base
    """
    # Initialize
    br_region = bedrock_agt_client.meta.region_name
    ds_id = ''
    s3_bucket_name = ''
    aoss_collection_arn = ''
    # Check if a KB id has been specified already
    if len(kb_id) == 0:
        # If a KB has not been specified, then, attempt to retrieve the first available active KB
        kb_found = False
        logging.info(
            "No Knowledge Bases (KBs) specified for use. Will attempt to retrieve the first available KB that meets all requirements.")
        # Get all the KBs
        list_kbs_response = bedrock_agt_client.list_knowledge_bases(maxResults=1)
        kbs = list_kbs_response['knowledgeBaseSummaries']
        # Loop through the KBs
        for kb in kbs:
            kb_id = kb['knowledgeBaseId']
            kb_meets_reqs, kb_name, ds_id, s3_bucket_name, embedding_model_arn, aoss_collection_arn, vector_ingestion_config \
                = does_kb_meet_requirements(bedrock_agt_client, kb_id)
            if kb_meets_reqs:
                logging.info(
                    "Found a Knowledge Base (KB) with id '{}' that meets all requirements. This KB will be used."
                    .format(kb_id))
                logging.info("Associated data source id is '{}'.".format(ds_id))
                logging.info("Associated S3 bucket is '{}'.".format(s3_bucket_name))
                logging.info("Associated Embedding model is '{}'.".format(embedding_model_arn))
                logging.info("Associated AOSS collection is '{}'.".format(aoss_collection_arn))
                logging.info("Associated vector ingestion config is '{}'.".format(vector_ingestion_config))
                logging.info("For more info on this KB, visit https://{}.console.aws.amazon.com/bedrock/home?region={}#/knowledge-bases/{}/{}/1"
                             .format(br_region, br_region, kb_name, kb_id))
                break
        # If an active KB is still not found, then print info for the user to create a KB
        if len(kb_id) == 0:
            logging.error("No Knowledge Base that meets all requirements was found.")
            logging.info(
                "Refer to the requirements specified earlier and use the process described at {} to create a Knowledge Base in the same AWS Region as Amazon Bedrock and rerun this cell."
                .format("https://docs.aws.amazon.com/bedrock/latest/userguide/knowledge-base-create.html"))
    else:
        # If a KB has not been specified, then, check if meets all the requirements
        kb_meets_reqs, kb_name, ds_id, s3_bucket_name, embedding_model_arn, aoss_collection_arn, vector_ingestion_config \
            = does_kb_meet_requirements(bedrock_agt_client, kb_id)
        if kb_meets_reqs:
            logging.info("The specified Knowledge Base (KB) with id '{}' meets all requirements. This KB will be used."
                         .format(kb_id))
            logging.info("Associated data source id is '{}'.".format(ds_id))
            logging.info("Associated S3 bucket is '{}'.".format(s3_bucket_name))
            logging.info("Associated Embedding model is '{}'.".format(embedding_model_arn))
            logging.info("Associated AOSS collection is '{}'.".format(aoss_collection_arn))
            logging.info("Associated vector ingestion config is '{}'.".format(vector_ingestion_config))
            logging.info("For more info on this KB, visit https://{}.console.aws.amazon.com/bedrock/home?region={}#/knowledge-bases/{}/{}/1"
                         .format(br_region, br_region, kb_name, kb_id))
        else:
            logging.error(
                "The specified Knowledge Base (KB) with id '{}' does not meet all requirements.".format(kb_id))
            logging.info(
                "Refer to the requirements specified earlier and use the process described at {} to create a Knowledge Base in the same AWS Region as Amazon Bedrock and rerun this cell."
                .format("https://docs.aws.amazon.com/bedrock/latest/userguide/knowledge-base-create.html"))
    return kb_id, ds_id, s3_bucket_name, aoss_collection_arn


def does_kb_meet_requirements(bedrock_agt_client, kb_id):
    """
    Function to evaluate of the specified KB meets all the requirements for the notebook

    Parameters:
    bedrock_agt_client (boto3 client): The boto3 client for Bedrock Agent
    kb_id (string): The id of the Knowledge Base

    Returns:
    boolean: The flag that specifies if the specified Knowledge Base meets requirements or not
    string: The name of the Knowledge Base
    string: The id of the data source associated with the above Knowledge Base
    string: The name of the S3 bucket that is associated with the above data source
    string: The ARN of the Embeddings Model associated with the above Knowledge Base
    string: The ARN of the Amazon OpenSearch Serverless (AOSS) collection associated with the above Knowledge Base
    dict: The vector ingestion configuration associated with the above Knowledge Base
    """
    kb_meets_reqs = False
    ds_id = ''
    kb_name = ''
    s3_bucket_name = ''
    embedding_model_arn = ''
    aoss_collection_arn = ''
    vector_ingestion_config = None
    # Get the details of the specified Knowledge Base
    get_kb_details_response = bedrock_agt_client.get_knowledge_base(knowledgeBaseId=kb_id)
    kb_vector_index_type = get_kb_details_response['knowledgeBase']['storageConfiguration']['type']
    # Check if Amazon OpenSearch Serverless
    if kb_vector_index_type == 'OPENSEARCH_SERVERLESS':
        # Get all data sources associated with this KB
        list_dss_response = bedrock_agt_client.list_data_sources(knowledgeBaseId=kb_id,
                                                                 maxResults=1)
        dss = list_dss_response['dataSourceSummaries']
        # Loop through the data sources
        for ds in dss:
            ds_id = ds['dataSourceId']
            ds_status = ds['status']
            # Check for active status
            if ds_status == 'AVAILABLE':
                get_ds_details_response = bedrock_agt_client.get_data_source(dataSourceId=ds_id,
                                                                             knowledgeBaseId=kb_id)
                ds_type = get_ds_details_response['dataSource']['dataSourceConfiguration']['type']
                if ds_type == 'S3':
                    kb_name = get_kb_details_response['knowledgeBase']['name']
                    s3_bucket_arn = get_ds_details_response['dataSource']['dataSourceConfiguration']['s3Configuration'][
                        'bucketArn']
                    # Helper function available through ./scripts/helper_functions.py
                    s3_bucket_name = get_s3_bucket_name_from_arn(s3_bucket_arn)
                    embedding_model_arn = get_kb_details_response['knowledgeBase']['knowledgeBaseConfiguration']['vectorKnowledgeBaseConfiguration']['embeddingModelArn']
                    aoss_collection_arn = get_kb_details_response['knowledgeBase']['storageConfiguration']['opensearchServerlessConfiguration']['collectionArn']
                    if 'vectorIngestionConfiguration' in get_ds_details_response['dataSource']:
                        vector_ingestion_config = str(get_ds_details_response['dataSource']['vectorIngestionConfiguration'])
                    kb_meets_reqs = True
                    break
    return kb_meets_reqs, kb_name, ds_id, s3_bucket_name, embedding_model_arn, aoss_collection_arn, vector_ingestion_config


def sync_to_kb(bedrock_agt_client, ds_id, kb_id, job_desc):
    """
    Function to sync data from the specified data source into the specified KB

    Parameters:
    bedrock_agt_client (boto3 client): The boto3 client for Bedrock Agent
    ds_id (string): The id of the data source
    kb_id (string): The id of the Knowledge Base
    job_desc (string): The description for this sync job

    Returns:
    None
    """
    start_ingestion_job_response = bedrock_agt_client.start_ingestion_job(dataSourceId=ds_id,
                                                                          description=job_desc,
                                                                          knowledgeBaseId=kb_id)
    ingestion_job_id = start_ingestion_job_response['ingestionJob']['ingestionJobId']
    logging.info("Ingestion job '{}' started.'".format(ingestion_job_id))
    # Sleep every 5 seconds; retrieve and check the status of the ingestion job
    # until it goes to COMPLETE or FAILED state
    while True:
        get_ingestion_job_response = bedrock_agt_client.get_ingestion_job(dataSourceId=ds_id,
                                                                          ingestionJobId=ingestion_job_id,
                                                                          knowledgeBaseId=kb_id)
        ingestion_job_status = get_ingestion_job_response['ingestionJob']['status']
        logging.info("Ingestion job '{}' is in '{}' status.".format(ingestion_job_id, ingestion_job_status))
        if ingestion_job_status in {'COMPLETE', 'FAILED'}:
            break
        else:
            logging.info("Waiting for 5 seconds to check the status...")
        time.sleep(5)


def retrieve_from_kb_using_boto3(bedrock_agt_rt_client, kb_id, query, max_results):
    """
    Function to retrieve data that matches the specified query from the specified KB using the boto3 API

    Parameters:
    bedrock_agt_rt_client (boto3 client): The boto3 client for Bedrock Agent Runtime
    kb_id (string): The id of the Knowledge Base
    query (string): The query to search for in the Knowledge Base
    max_results (int): The max number of results to return from the search

    Returns:
    list: A list of query result dict objects
    """
    query_results = []
    retrieve_response = bedrock_agt_rt_client.retrieve(
        knowledgeBaseId=kb_id,
        retrievalConfiguration={
            'vectorSearchConfiguration': {
                'numberOfResults': max_results,
                'overrideSearchType': 'SEMANTIC'
            }
        },
        retrievalQuery={
            'text': query
        }
    )
    retrieve_query_results = retrieve_response['retrievalResults']
    for retrieve_query_result in retrieve_query_results:
        query_results.append(
            {
                "query_result": retrieve_query_result['content']['text'],
                "relevancy_score": 0.0
            }
        )
    return query_results


def retrieve_from_kb_using_lc(bedrock_agt_rt_client, kb_id, query, max_results):
    """
    Function to retrieve data that matches the specified query from the specified KB using the LangChain API

    Parameters:
    bedrock_agt_rt_client (boto3 client): The boto3 client for Bedrock Agent Runtime
    kb_id (string): The id of the Knowledge Base
    query (string): The query to search for in the Knowledge Base
    max_results (int): The max number of results to return from the search

    Returns:
    list: A list of query result dict objects
    """
    query_results = []
    retriever = AmazonKnowledgeBasesRetriever(
        knowledge_base_id=kb_id,
        client=bedrock_agt_rt_client,
        retrieval_config={"vectorSearchConfiguration": {"numberOfResults": max_results}}
    )
    retrieve_query_results = retriever.invoke(input=query)
    for retrieve_query_result in retrieve_query_results:
        query_results.append(
            {
                "query_result": retrieve_query_result.page_content,
                "relevancy_score": 0.0
            }
        )
    return query_results


def delete_kb(bedrock_agt_client, kb_id):
    """
    Function to delete the specified KB

    Parameters:
    bedrock_agt_client (boto3 client): The boto3 client for Bedrock Agent
    kb_id (string): The id of the Knowledge Base

    Returns:
    None
    """
    logging.info("Initiated the delete of Knowledge Base '{}'.".format(kb_id))
    delete_kb_response = bedrock_agt_client.delete_knowledge_base(
        knowledgeBaseId=kb_id
    )
    logging.info("Status is '{}'.".format(delete_kb_response['status']))


def get_br_agent_that_meets_requirements(bedrock_agt_client, br_agent_id, br_agent_version):
    """
    Function to get the first available Bedrock agent that meets the requirements

    Parameters:
    bedrock_agt_client (boto3 client): The boto3 client for Bedrock Agent
    br_agent_id (string): The id of the Bedrock Agent

    Returns:
    string: The id of the Bedrock Agent
    string: The version of the Bedrock Agent
    """
    # Initialize
    br_region = bedrock_agt_client.meta.region_name
    # Check if an agent id has been specified already
    if len(br_agent_id) == 0:
        # If an agent has not been specified, then, attempt to retrieve the first available active agent
        logging.info(
            "No Amazon Bedrock Agents specified for use. Will attempt to retrieve the first available agent that meets all requirements.")
        # Get the list of agents
        list_agents_response = bedrock_agt_client.list_agents(
            maxResults=100
        )
        # Loop through the agent list
        agent_summaries = list_agents_response['agentSummaries']
        for agent_summary in agent_summaries:
            # Check the prepared status and get the agent id
            if agent_summary['agentStatus'] == 'PREPARED':
                br_agent_id = agent_summary['agentId']
                br_agent_name = agent_summary['agentName']
                br_agent_version = agent_summary['latestAgentVersion']
                logging.info("Found an Amazon Bedrock Agent that meets all requirements. It's id is '{}', it's name is '{}' and it's version is '{}'.".
                             format(br_agent_id, br_agent_name, br_agent_version))
                logging.info(
                    "For more info on this Amazon Bedrock Agent, visit https://{}.console.aws.amazon.com/bedrock/home?region={}#/agents/{}"
                    .format(br_region, br_region, br_agent_id))
                break
        # Check if the agent id exists
        if len(br_agent_id) == 0:
            # If an active agent is still not found, then print info for the user to create an agent
            logging.error("No Amazon Bedrock Agent that meets all requirements was found.")
            logging.info(
                "Refer to the requirements specified earlier and use the process described at {} to create an Amazon Bedrock agent in the same AWS Region as Amazon Bedrock and rerun this cell."
                .format("https://docs.aws.amazon.com/bedrock/latest/userguide/agents-create.html"))
    else:
        # If an agent has been specified, then, check if it meets all requirements
        get_agent_response = bedrock_agt_client.get_agent(
            agentId=br_agent_id
        )
        if get_agent_response['agent']['agentStatus'] == 'PREPARED':
            br_agent_name = get_agent_response['agent']['agentName']
            #br_agent_version = get_agent_response['agent']['agentVersion']
            logging.info("The specified Amazon Bedrock Agent with id '{}' meets all requirements. This agent will be used. It's name is '{}' and it's version is '{}'."
                         .format(br_agent_id, br_agent_name, br_agent_version))
            logging.info(
                "For more info on this Amazon Bedrock Agent, visit https://{}.console.aws.amazon.com/bedrock/home?region={}#/agents/{}"
                .format(br_region, br_region, br_agent_id))
        else:
            logging.error(
                "The specified Amazon Bedrock Agent with id '{}' does not meet all requirements.".format(br_agent_id))
            logging.info(
                "Refer to the requirements specified earlier and use the process described at {} to create an Amazon Bedrock agent in the same AWS Region as Amazon Bedrock and rerun this cell."
                .format("https://docs.aws.amazon.com/bedrock/latest/userguide/agents-create.html"))
    return br_agent_id, br_agent_version


def get_br_agent_alias_that_meets_requirements(bedrock_agt_client, br_agent_id, br_agent_alias_id):
    """
    Function to get the first available Bedrock agent alias that meets the requirements

    Parameters:
    bedrock_agt_client (boto3 client): The boto3 client for Bedrock Agent
    br_agent_id (string): The id of the Bedrock Agent
    br_agent_alias_id (string): The id of the Bedrock Agent Alias

    Returns:
    string: The id of the Bedrock Agent Alias
    """
    # Initialize
    br_region = bedrock_agt_client.meta.region_name
    # Check if an agent alias id has been specified already
    if len(br_agent_alias_id) == 0:
        # If an agent alias has not been specified, then, attempt to retrieve the first available active agent alias
        logging.info(
            "No Amazon Bedrock Agent aliases specified for use. Will attempt to retrieve the first available agent alias that meets all requirements.")
        # Get the list of agent aliases associated with this agent
        list_agent_aliases_response = bedrock_agt_client.list_agent_aliases(
            agentId=br_agent_id,
            maxResults=100
        )
        # Loop through the agent alias list
        agent_alias_summaries = list_agent_aliases_response['agentAliasSummaries']
        for agent_alias_summary in agent_alias_summaries:
            # Check the prepared status and get the agent alias id
            if agent_alias_summary['agentAliasStatus'] == 'PREPARED':
                br_agent_alias_id = agent_alias_summary['agentAliasId']
                br_agent_alias_name = agent_alias_summary['agentAliasName']
                logging.info("Found an Amazon Bedrock Agent alias that meets all requirements. It's id is '{}' and it's name is '{}'.".
                             format(br_agent_alias_id, br_agent_alias_name))
                logging.info(
                    "For more info on this Amazon Bedrock Agent alias, visit https://{}.console.aws.amazon.com/bedrock/home?region={}#/agents/{}/alias/{}"
                    .format(br_region, br_region, br_agent_id, br_agent_alias_id))
                break
        # Check if the agent alias id exists
        if len(br_agent_id) == 0:
            # If an active agent alias is still not found, then print info for the user to create an agent alias
            logging.error("No Amazon Bedrock Agent alias that meets all requirements was found.")
            logging.info(
                "Refer to the requirements specified earlier and use the process described at {} to create an Amazon Bedrock Agent alias in the same AWS Region as Amazon Bedrock and rerun this cell."
                .format("https://docs.aws.amazon.com/bedrock/latest/userguide/agents-alias-manage.html"))
    else:
        # If an agent alias has been specified, then, check if it meets all requirements
        get_agent_alias_response = bedrock_agt_client.get_agent_alias(
            agentAliasId=br_agent_alias_id,
            agentId=br_agent_id
        )
        if get_agent_alias_response['agentAlias']['agentAliasStatus'] == 'PREPARED':
            br_agent_alias_name = get_agent_alias_response['agentAlias']['agentAliasName']
            logging.info("The specified Amazon Bedrock Agent alias with id '{}' meets all requirements. This agent alias will be used. It's name is '{}'."
                         .format(br_agent_alias_id, br_agent_alias_name))
            logging.info(
                "For more info on this Amazon Bedrock Agent alias, visit https://{}.console.aws.amazon.com/bedrock/home?region={}#/agents/{}/alias/{}"
                .format(br_region, br_region, br_agent_id, br_agent_alias_id))
        else:
            logging.error("No Amazon Bedrock Agent aliases that meets all requirements was found.")
            logging.info(
                "Refer to the requirements specified earlier and use the process described at {} to create an Amazon Bedrock Agent alias in the same AWS Region as Amazon Bedrock and rerun this cell."
                .format("https://docs.aws.amazon.com/bedrock/latest/userguide/agents-alias-manage.html"))
    return br_agent_alias_id


def prepare_prompt(prompt_template_dir, prompt_template_file_name, **kwargs):
    """
    Function to prepare the prompt

    Parameters:
    prompt_template_dir (string): The directory that contains the prompt templates
    prompt_template_file_name (string): The name of the prompt template file
    **kwargs (**kwargs): The variable names and values to substitute in the prompt template

    Returns:
    string: The prepared prompt
    """
    prompt_template_file_path = os.path.join(prompt_template_dir, prompt_template_file_name)
    logging.info('Reading content from prompt template file "{}"...'.format(prompt_template_file_name))
    prompt_template = PromptTemplate.from_file(prompt_template_file_path)
    logging.info('Completed reading content from prompt template file.')
    logging.info('Substituting prompt variables...')
    prompt = prompt_template.format(**kwargs)
    logging.info('Completed substituting prompt variables.')
    return prompt


def invoke_claude_3(model_id, bedrock_rt_client, system_prompt, user_prompt, log_prompt_response):
    """
    Function to invoke the specified Claude 3 LLM through the LangChain ChatModel client and using the specified prompt

    Parameters:
    model_id (string): The id of the Anthropic Claude 3 LLM in Bedrock
    bedrock_rt_client (boto3 client): The boto3 client for Bedrock Runtime
    system_prompt (string): The system prompt to the LLM
    user_prompt (string): The user prompt to the LLM
    log_prompt_response (boolean): The flag to enable/disable logging of the prompt response

    Returns:
    string: The prompt response from the LLM
    """
    # Create the LangChain ChatModel client
    logging.debug('Creating LangChain ChatBedrock client for LLM "{}"...'.format(model_id))
    llm = ChatBedrock(
        model_id=model_id,
        model_kwargs={
            "temperature": 0,
            "max_tokens": 4000
        },
        client=bedrock_rt_client,
        streaming=False
    )
    logging.debug('Completed creating LangChain ChatBedrock client for LLM.')
    messages = [
        SystemMessage(
            content=system_prompt
        ),
        HumanMessage(
            content=user_prompt
        )
    ]
    logging.info('Invoking LLM "{}" with specified inference parameters "{}"...'.
                 format(llm.model_id, llm.model_kwargs))
    start = timer()
    prompt_response = llm.invoke(messages).content
    end = timer()
    if log_prompt_response:
        prompt = ''
        for message in messages:
            prompt += message.content + '\n\n'
        prompt = prompt.rstrip('\n')
        logging.info('PROMPT: {}'.format(prompt))
        logging.info('RESPONSE: {}'.format(prompt_response))
    logging.info('Completed invoking LLM.')
    logging.info('Prompt processing duration = {} second(s)'.format(end - start))
    return prompt_response


def instruct_llm_to_find_relevancy(model_id, bedrock_rt_client, prompt_templates_dir,
                                   system_prompt_template_file, user_prompt_template_file,
                                   query, query_result):
    """
    Function to instruct the specified LLM to find the relevancy between two specified pieces of text

    Parameters:
    model_id (string): The id of the Anthropic Claude 3 LLM in Bedrock
    bedrock_rt_client (boto3 client): The boto3 client for Bedrock Runtime
    prompt_templates_dir (string): The directory that contains the prompt templates
    system_prompt_template_file (string): The name of the system prompt template file
    user_prompt_template_file (string): The name of the user prompt template file
    query (string): The query from the user
    query_result (string): One of the query results retrieved from the Knowledge Base

    Returns:
    float: The relevancy score of the query result to the query
    """
    # Read the prompt template and perform variable substitution
    system_prompt = prepare_prompt(prompt_templates_dir,
                                   system_prompt_template_file)
    user_prompt = prepare_prompt(prompt_templates_dir,
                                 user_prompt_template_file,
                                 QUERY=query,
                                 QUERY_RESULT=query_result)
    # Invoke the LLM and return the response
    return float(invoke_claude_3(model_id, bedrock_rt_client, system_prompt, user_prompt, False))


def process_final_prompt(model_id, bedrock_rt_client, prompt_templates_dir,
                         system_prompt_template_file, user_prompt_template_file,
                         query, query_results):
    """
    Function that processes the final prompt and invokes the LLM

    Parameters:
    model_id (string): The id of the Anthropic Claude 3 LLM in Bedrock
    bedrock_rt_client (boto3 client): The boto3 client for Bedrock Runtime
    prompt_templates_dir (string): The directory that contains the prompt templates
    system_prompt_template_file (string): The name of the system prompt template file
    user_prompt_template_file (string): The name of the user prompt template file
    query (string): The query from the user enhanced with the context from the Knowledge Base
    query_results (list): The list of query result dict objects

    Returns:
    string: The prompt response from the LLM
    """
    context = ''
    for query_result in query_results:
        context += query_result['query_result']
        context += '\n\n'
    # Read the prompt template and perform variable substitution
    system_prompt = prepare_prompt(prompt_templates_dir,
                                   system_prompt_template_file)
    user_prompt = prepare_prompt(prompt_templates_dir,
                                 user_prompt_template_file,
                                 CONTEXT=context,
                                 QUERY=query)
    # Invoke the LLM and return the response
    return invoke_claude_3(model_id, bedrock_rt_client, system_prompt, user_prompt, True)


def perform_web_search(br_agent_alias_id, br_agent_id, bedrock_agt_rt_client, query):
    """
    Function that processes the web search using Agents for Amazon Bedrock

    Parameters:
    br_agent_alias_id (string): The id of the Bedrock Agent Alias
    br_agent_id (string): The id of the Bedrock Agent
    bedrock_agt_rt_client (boto3 client): The boto3 client for Bedrock Agent Runtime
    query (string): The query from the user

    Returns:
    string: The final response from the LLM with the prompt containing the results of the web search
    """
    # Invoke the Bedrock Agent
    logging.info('Invoking Bedrock Agent with id "{}" and alias "{}"...'.format(br_agent_id, br_agent_alias_id))
    start = timer()
    invoke_agent_response = bedrock_agt_rt_client.invoke_agent(
        agentAliasId=br_agent_alias_id,
        agentId=br_agent_id,
        enableTrace=True,
        endSession=False,
        inputText=query,
        sessionId=str(uuid.uuid4())
    )
    completion = ''
    # Process the response
    for event in invoke_agent_response.get('completion'):
        logging.debug(event)
        # Check if the 'chunk' object exists and read it
        if 'chunk' in event:
            chunk = event['chunk']
            completion += chunk['bytes'].decode()
    end = timer()
    logging.info('Completed invoking Bedrock Agent.')
    logging.info('Prompt processing duration = {} second(s)'.format(end - start))
    return completion
