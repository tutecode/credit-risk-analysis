### Guide to Download dataset

1. To access the dataset you will have to use the boto3 library and here are the key and secret key that you can use for read-only access.

2. install aws cli, using this guide [aws cli install](https://docs.aws.amazon.com/es_es/cli/v1/userguide/cli-chap-install.html)

3. then configure aws with "key" and "secret" using the command aws configure

4. You'll be prompted to enter your access key, secret access key, default region, and output format. in region and output only with None

5. Finally use copy command from aws to local directory
aws s3 cp <s3_url> <local_destination>

**For example using this code:**

`bash
aws s3 cp --recursive s3://mys3url/ ./local-folder/
`

* you will have this structure in your folder:
  - LeaderBoard_Data.zip
  - Leaderboard_Submission_Example.zip
  - PAKDD2010_Leaderboard_Submission_Example.txt
  - PAKDD2010_Modeling_Data.txt
  - PAKDD2010_Prediction_Data.txt
  - 'PAKDD-2010 training data.zip'
  - PAKDD2010_VariablesList.XLS
  - Prediction_Data.zip


