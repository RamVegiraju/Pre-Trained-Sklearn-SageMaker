import boto3
import json

runtime_client = boto3.client('sagemaker-runtime')
content_type = "application/json"
request_body = {"Input": [[0.09178, 0.0, 4.05, 0.0, 0.51, 6.416, 84.1, 2.6463, 5.0, 296.0, 16.6, 395.5, 9.04]]}
data = json.loads(json.dumps(request_body))
payload = json.dumps(data)
endpoint_name = "enter ep name here"

response = runtime_client.invoke_endpoint(
    EndpointName=endpoint_name,
    ContentType=content_type,
    Body=payload)
result = json.loads(response['Body'].read().decode())['Output']
print(result)