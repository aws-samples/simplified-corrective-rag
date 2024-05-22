## Build a simplified Corrective RAG assistant with Amazon Bedrock

### Overview

[Amazon Bedrock](https://aws.amazon.com/bedrock/) is a fully managed service that makes high-performing foundation models (FMs) from leading AI startups and Amazon available for your use through a unified API. You can choose from a wide range of foundation models to find the model that is best suited for your use case. Amazon Bedrock also offers a broad set of capabilities to build generative AI applications with security, privacy, and responsible AI. Using Amazon Bedrock, you can easily experiment with and evaluate top foundation models for your use cases, privately customize them with your data using techniques such as fine-tuning and Retrieval Augmented Generation (RAG), and build agents that execute tasks using your enterprise systems and data sources.

[Large Language Models (LLMs)](https://en.wikipedia.org/wiki/Large_language_model) inevitably exhibit hallucinations since the accuracy of generated texts cannot be secured solely by the parametric knowledge they encapsulate. Although Retrieval Augmented Generation (RAG) is a practicable complement to LLMs, it relies heavily on the relevance of retrieved documents, raising concerns about how the model behaves if retrieval goes wrong.

Advanced RAG techniques like [Corrective RAG](https://arxiv.org/pdf/2401.15884.pdf) were proposed to improve the robustness of generation. In CRAG, a lightweight retrieval evaluator is designed to assess the overall quality of retrieved documents for a query, returning a confidence degree based on which different knowledge retrieval actions can be triggered. Since retrieval from static and limited corpora can only return sub-optimal documents, large-scale web searches are utilized as an extension for augmenting the retrieval results. CRAG is plug-and-play and can be seamlessly coupled with various RAG-based approaches.

This repository contains code that will walk you through the process of building a simplified CRAG based assistant. We will cover two scenarios for the retrieval phase:

1. **(Scenario 1)** A document that closely matches the specified query is located in the Knowledge Base.
2. **(Scenario 2)** A document that closely matches the specified query is not located in the Knowledge Base. As a result, a web search will be performed to retrieve matching document(s).

### To get started

1. Choose an AWS Account to use and make sure to create all resources in that Account.
2. Identify an AWS Region that has [Amazon Bedrock with Anthropic Claude 3 and Titan Embeddings G1 - Text](https://docs.aws.amazon.com/bedrock/latest/userguide/models-regions.html) models.
3. In that Region, copy the following file to a new or existing [Amazon S3 bucket](https://docs.aws.amazon.com/AmazonS3/latest/userguide/UsingBucket.html) of your choice. Make sure that this bucket can be read by [AWS CloudFormation](https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/Welcome.html).
   * [wikipedia_search_bedrock_agent.zip](https://github.com/aws-samples/simplified-corrective-rag/blob/main/assets/dependencies/wikipedia_search_bedrock_agent.zip)
4. Create the Lambda layer file named `py312_opensearch-py_requests_and_requests-aws4auth.zip` using the following procedure and upload it to the same Amazon S3 bucket as in step 3.
   - On Windows 10 or above:
     1. Make sure [Python 3.12](https://www.python.org/downloads/release/python-3120/) and [pip](https://pip.pypa.io/en/stable/installation/) are installed and set in the user's PATH variable.
     2. Download [7-zip](https://www.7-zip.org/) and install it in `C:/Program Files/7-Zip/`.
     3. Open the Windows command prompt.
     4. Create a new directory and `cd` into it.
     5. Run the [lambda_layer_file_create.bat](https://github.com/aws-samples/simplified-corrective-rag/blob/main/assets/dependencies/lambda_layer_file_create.bat) from inside of that directory.
     6. This will create the Lambda layer file named `py312_opensearch-py_requests_and_requests-aws4auth.zip`.
   - On Linux:
     1. Make sure [Python 3.12](https://www.python.org/downloads/release/python-3120/) and [pip](https://pip.pypa.io/en/stable/installation/) are installed and set in the user's PATH variable.
     2. Open the Linux command prompt.
     3. Create a new directory and `cd` into it.
     4. Run the [lambda_layer_file_create.sh](https://github.com/aws-samples/simplified-corrective-rag/blob/main/assets/dependencies/lambda_layer_file_create.sh) from inside of that directory.
     5. This will create the Lambda layer file named `py312_opensearch-py_requests_and_requests-aws4auth.zip`.
5. Take the provided AWS CloudFormation template [simplified-corrective-rag-cfn.yaml](https://github.com/aws-samples/simplified-corrective-rag/blob/main/assets/simplified-corrective-rag-cfn.yaml) and update the following parameter,
   * *DeploymentArtifactsS3BucketName* - set this to the name of the Amazon S3 bucket from step 3.
6. Create an [AWS CloudFormation stack](https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/cfn-whatis-concepts.html#cfn-concepts-stacks) with the updated template.
7. Open the Jupyter notebook named *simplified-corrective-rag.ipynb* by navigating to the [Amazon SageMaker notebook instances console](https://docs.aws.amazon.com/sagemaker/latest/dg/howitworks-access-ws.html) and clicking on the *Open Jupyter* link on the instance named *simplified-crag-instance*.

### Repository structure

* [An assets folder](https://github.com/aws-samples/simplified-corrective-rag/blob/main/assets) that contains the AWS CloudFormation template and the dependent  artifacts.
* [The Python code for an AWS Lambda function](https://github.com/aws-samples/simplified-corrective-rag/blob/main/lambda) that will be invoked by the Bedrock Agent to perform the web search. This is also zipped into [this file](https://github.com/aws-samples/simplified-corrective-rag/blob/main/assets/dependencies/wikipedia_search_bedrock_agent.zip) as a dependent artifact.
* [A notebooks folder](https://github.com/aws-samples/simplified-corrective-rag/blob/main/notebooks) that contains all the artifacts related to the Jupyter notebook that you will be working on.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.

