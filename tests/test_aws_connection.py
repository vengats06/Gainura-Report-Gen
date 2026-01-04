import boto3
from backend.config import Config

# Test S3 connection
s3 = boto3.client(
    's3',
    aws_access_key_id=Config.AWS_ACCESS_KEY_ID,
    aws_secret_access_key=Config.AWS_SECRET_ACCESS_KEY,
    region_name=Config.AWS_REGION
)

# List buckets
response = s3.list_buckets()
print("✅ S3 Connection Successful!")
print("Your buckets:")
for bucket in response['Buckets']:
    print(f"  - {bucket['Name']}")

# Test RDS connection
import psycopg2
conn = psycopg2.connect(
    host=Config.AWS_RDS_HOST,
    port=Config.AWS_RDS_PORT,
    database=Config.AWS_RDS_DATABASE,
    user=Config.AWS_RDS_USER,
    password=Config.AWS_RDS_PASSWORD
)
print("✅ RDS Connection Successful!")
conn.close()