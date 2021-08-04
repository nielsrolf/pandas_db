# Deploy to EC2
Create a ec2-instance (ubuntu), ssh into it, configure it as described below and run `source /home/ubuntu/pandas_db/ec2/setup_ec2.sh`.
Edit the inbound rules of the security group to allow traffic on port 8050, and you should be able to access the UI via <public ec2 address>:8050

Stuff that needs to be configured manually on ec2:
- `export PANDAS_DB_S3_PREFIX="https://<your s3 bucket with anb upload of $PANDAS_DB_PATH>.s3.eu-central-1.amazonaws.com/"` goes to .bashrc
