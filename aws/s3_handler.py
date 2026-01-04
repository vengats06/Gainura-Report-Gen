"""
AWS S3 Handler - Upload/Download Files to Amazon S3
===================================================

This module handles all interactions with AWS S3 buckets.
S3 (Simple Storage Service) is like Google Drive but for applications.

Key Features:
- Upload files (JSON, CSV, Parquet, PNG, PDF)
- Download files
- List files in bucket
- Delete files
- Check if file exists
- Generate presigned URLs (temporary download links)

Usage:
    from aws.s3_handler import S3Handler
    s3 = S3Handler()
    s3.upload_json(data, 'angel_one/TCS_2024-12-17.json')
"""

import boto3
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from botocore.exceptions import ClientError, NoCredentialsError
from backend.config import Config
from utils.logger import get_logger

logger = get_logger(__name__)


class S3Handler:
    """
    AWS S3 Operations Handler
    
    This class provides methods to interact with S3 buckets.
    All methods handle errors gracefully and log operations.
    """
    
    def __init__(self):
        """
        Initialize S3 client with credentials from Config.
        
        The boto3 client automatically reads AWS credentials from:
        1. Environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
        2. ~/.aws/credentials file
        3. IAM role (when running on AWS services)
        
        We use explicit credentials from Config for clarity.
        """
        try:
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=Config.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=Config.AWS_SECRET_ACCESS_KEY,
                region_name=Config.AWS_REGION
            )
            
            self.raw_bucket = Config.AWS_S3_BUCKET_RAW
            self.processed_bucket = Config.AWS_S3_BUCKET_PROCESSED
            
            logger.info("S3Handler initialized successfully")
            logger.info(f"Raw bucket: {self.raw_bucket}")
            logger.info(f"Processed bucket: {self.processed_bucket}")
            
        except NoCredentialsError:
            logger.error("AWS credentials not found!")
            raise Exception("AWS credentials missing. Check .env file.")
        except Exception as e:
            logger.error(f"Failed to initialize S3Handler: {str(e)}")
            raise
    
    
    def upload_file(self, local_path: str, s3_key: str, bucket_type: str = 'raw') -> bool:
        """
        Upload a file from local system to S3.
        
        Args:
            local_path (str): Path to local file (e.g., 'temp/data.json')
            s3_key (str): S3 object key (e.g., 'angel_one/TCS_2024-12-17.json')
            bucket_type (str): 'raw' or 'processed'
            
        Returns:
            bool: True if successful, False otherwise
            
        Example:
            s3.upload_file('temp/TCS.json', 'angel_one/TCS_2024-12-17.json')
            # File is now at: s3://gainura-raw-data/angel_one/TCS_2024-12-17.json
        """
        bucket = self.raw_bucket if bucket_type == 'raw' else self.processed_bucket
        
        try:
            logger.info(f"Uploading {local_path} to s3://{bucket}/{s3_key}")
            
            # Upload file
            self.s3_client.upload_file(local_path, bucket, s3_key)
            
            logger.info(f"Successfully uploaded to S3: {s3_key}")
            return True
            
        except FileNotFoundError:
            logger.error(f" Local file not found: {local_path}")
            return False
        except ClientError as e:
            logger.error(f" S3 upload failed: {str(e)}")
            return False
    
    
    def upload_json(self, data: Dict, s3_key: str, bucket_type: str = 'raw') -> bool:
        """
        Upload JSON data directly to S3 (without saving to local file first).
        
        Args:
            data (Dict): Python dictionary to upload
            s3_key (str): S3 object key
            bucket_type (str): 'raw' or 'processed'
            
        Returns:
            bool: True if successful, False otherwise
            
        Example:
            data = {"symbol": "TCS", "price": 3842.50}
            s3.upload_json(data, 'angel_one/TCS.json')
        """
        bucket = self.raw_bucket if bucket_type == 'raw' else self.processed_bucket
        
        try:
            logger.info(f"Uploading JSON to s3://{bucket}/{s3_key}")
            
            # Convert dict to JSON string
            json_string = json.dumps(data, indent=2, default=str)
            
            # Upload directly to S3
            self.s3_client.put_object(
                Bucket=bucket,
                Key=s3_key,
                Body=json_string,
                ContentType='application/json'
            )
            
            logger.info(f" Successfully uploaded JSON to S3: {s3_key}")
            return True
            
        except Exception as e:
            logger.error(f" Failed to upload JSON: {str(e)}")
            return False
    
    
    def download_file(self, s3_key: str, local_path: str, bucket_type: str = 'raw') -> bool:
        """
        Download a file from S3 to local system.
        
        Args:
            s3_key (str): S3 object key
            local_path (str): Where to save locally
            bucket_type (str): 'raw' or 'processed'
            
        Returns:
            bool: True if successful, False otherwise
            
        Example:
            s3.download_file('angel_one/TCS.json', 'temp/TCS_downloaded.json')
        """
        bucket = self.raw_bucket if bucket_type == 'raw' else self.processed_bucket
        
        try:
            logger.info(f"Downloading s3://{bucket}/{s3_key} to {local_path}")
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            # Download file
            self.s3_client.download_file(bucket, s3_key, local_path)
            
            logger.info(f"Successfully downloaded from S3: {s3_key}")
            return True
            
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                logger.error(f"File not found in S3: {s3_key}")
            else:
                logger.error(f" S3 download failed: {str(e)}")
            return False
    
    
    def download_json(self, s3_key: str, bucket_type: str = 'raw') -> Optional[Dict]:
        """
        Download and parse JSON file from S3 (without saving to disk).
        
        Args:
            s3_key (str): S3 object key
            bucket_type (str): 'raw' or 'processed'
            
        Returns:
            Dict or None: Parsed JSON data, or None if failed
            
        Example:
            data = s3.download_json('angel_one/TCS.json')
            print(data['symbol'])  # Output: TCS
        """
        bucket = self.raw_bucket if bucket_type == 'raw' else self.processed_bucket
        
        try:
            logger.info(f"Downloading JSON from s3://{bucket}/{s3_key}")
            
            # Get object from S3
            response = self.s3_client.get_object(Bucket=bucket, Key=s3_key)
            
            # Read and parse JSON
            json_content = response['Body'].read().decode('utf-8')
            data = json.loads(json_content)
            
            logger.info(f"Successfully downloaded JSON from S3: {s3_key}")
            return data
            
        except ClientError as e:
            logger.error(f" Failed to download JSON: {str(e)}")
            return None
    
    
    def list_files(self, prefix: str = '', bucket_type: str = 'raw') -> List[str]:
        """
        List all files in S3 bucket with given prefix.
        
        Args:
            prefix (str): Filter files by prefix (e.g., 'angel_one/')
            bucket_type (str): 'raw' or 'processed'
            
        Returns:
            List[str]: List of S3 keys
            
        Example:
            files = s3.list_files('angel_one/')
            # Output: ['angel_one/TCS.json', 'angel_one/RELIANCE.json', ...]
        """
        bucket = self.raw_bucket if bucket_type == 'raw' else self.processed_bucket
        
        try:
            logger.info(f"Listing files in s3://{bucket}/{prefix}")
            
            response = self.s3_client.list_objects_v2(
                Bucket=bucket,
                Prefix=prefix
            )
            
            if 'Contents' not in response:
                logger.info(f"No files found with prefix: {prefix}")
                return []
            
            files = [obj['Key'] for obj in response['Contents']]
            logger.info(f"Found {len(files)} files")
            
            return files
            
        except ClientError as e:
            logger.error(f"❌ Failed to list files: {str(e)}")
            return []
    
    
    def file_exists(self, s3_key: str, bucket_type: str = 'raw') -> bool:
        """
        Check if a file exists in S3.
        
        Args:
            s3_key (str): S3 object key
            bucket_type (str): 'raw' or 'processed'
            
        Returns:
            bool: True if file exists, False otherwise
            
        Example:
            if s3.file_exists('angel_one/TCS.json'):
                print("File already exists, skipping upload")
        """
        bucket = self.raw_bucket if bucket_type == 'raw' else self.processed_bucket
        
        try:
            self.s3_client.head_object(Bucket=bucket, Key=s3_key)
            logger.info(f"File exists: s3://{bucket}/{s3_key}")
            return True
            
        except ClientError:
            logger.info(f"File does not exist: s3://{bucket}/{s3_key}")
            return False
    
    
    def delete_file(self, s3_key: str, bucket_type: str = 'raw') -> bool:
        """
        Delete a file from S3.
        
        Args:
            s3_key (str): S3 object key
            bucket_type (str): 'raw' or 'processed'
            
        Returns:
            bool: True if successful, False otherwise
            
        Example:
            s3.delete_file('angel_one/old_data.json')
        """
        bucket = self.raw_bucket if bucket_type == 'raw' else self.processed_bucket
        
        try:
            logger.info(f"Deleting s3://{bucket}/{s3_key}")
            
            self.s3_client.delete_object(Bucket=bucket, Key=s3_key)
            
            logger.info(f" Successfully deleted from S3: {s3_key}")
            return True
            
        except ClientError as e:
            logger.error(f" Failed to delete file: {str(e)}")
            return False
    
    
    def generate_presigned_url(self, s3_key: str, bucket_type: str = 'raw', 
                               expiration: int = 3600) -> Optional[str]:
        """
        Generate a temporary download URL for a file in S3.
        
        This URL allows anyone with the link to download the file,
        but only for a limited time (default: 1 hour).
        
        Args:
            s3_key (str): S3 object key
            bucket_type (str): 'raw' or 'processed'
            expiration (int): URL validity in seconds (default: 3600 = 1 hour)
            
        Returns:
            str or None: Pre-signed URL, or None if failed
            
        Example:
            url = s3.generate_presigned_url('reports/TCS_Report.pdf')
            # User can download PDF by visiting this URL
            # URL expires after 1 hour
        """
        bucket = self.raw_bucket if bucket_type == 'raw' else self.processed_bucket
        
        try:
            logger.info(f"Generating presigned URL for s3://{bucket}/{s3_key}")
            
            url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': bucket, 'Key': s3_key},
                ExpiresIn=expiration
            )
            
            logger.info(f"Generated presigned URL (expires in {expiration}s)")
            return url
            
        except ClientError as e:
            logger.error(f" Failed to generate presigned URL: {str(e)}")
            return None
    
    
    def get_file_metadata(self, s3_key: str, bucket_type: str = 'raw') -> Optional[Dict]:
        """
        Get metadata about a file in S3 (size, last modified, etc.).
        
        Args:
            s3_key (str): S3 object key
            bucket_type (str): 'raw' or 'processed'
            
        Returns:
            Dict or None: File metadata
            
        Example:
            metadata = s3.get_file_metadata('angel_one/TCS.json')
            print(f"File size: {metadata['size']} bytes")
            print(f"Last modified: {metadata['last_modified']}")
        """
        bucket = self.raw_bucket if bucket_type == 'raw' else self.processed_bucket
        
        try:
            response = self.s3_client.head_object(Bucket=bucket, Key=s3_key)
            
            metadata = {
                'size': response['ContentLength'],
                'last_modified': response['LastModified'],
                'content_type': response.get('ContentType', 'unknown'),
                'etag': response['ETag']
            }
            
            logger.info(f" Retrieved metadata for: {s3_key}")
            return metadata
            
        except ClientError as e:
            logger.error(f" Failed to get metadata: {str(e)}")
            return None
    
    
    def create_folder_structure(self):
        """
        Create folder structure in S3 buckets.
        
        S3 doesn't have "folders", but we can create empty objects
        with trailing slashes to simulate folder structure.
        
        This is useful for organization and navigation in AWS Console.
        """
        folders = {
            'raw': [
                'angel_one/',
                'fundamentals/',
                'news/'
            ],
            'processed': [
                'technical_indicators/',
                'ml_predictions/',
                'charts/',
                'reports/'
            ]
        }
        
        for bucket_type, folder_list in folders.items():
            bucket = self.raw_bucket if bucket_type == 'raw' else self.processed_bucket
            
            for folder in folder_list:
                try:
                    self.s3_client.put_object(Bucket=bucket, Key=folder)
                    logger.info(f" Created folder: s3://{bucket}/{folder}")
                except ClientError as e:
                    logger.warning(f"Could not create folder {folder}: {str(e)}")


# Example usage and testing
if __name__ == "__main__":
    """
    Test S3Handler functionality
    Run this file directly to test S3 operations
    """
    print("\n" + "="*60)
    print("Testing S3Handler")
    print("="*60 + "\n")
    
    try:
        # Initialize S3 handler
        s3 = S3Handler()
        
        # Test 1: Upload JSON
        print("Test 1: Uploading JSON data...")
        test_data = {
            "symbol": "TCS",
            "price": 3842.50,
            "timestamp": datetime.now().isoformat()
        }
        success = s3.upload_json(test_data, 'test/sample.json')
        print(f"Upload JSON: {'Success' if success else '❌ Failed'}\n")
        
        # Test 2: Download JSON
        print("Test 2: Downloading JSON data...")
        downloaded_data = s3.download_json('test/sample.json')
        if downloaded_data:
            print(f"Downloaded data: {downloaded_data}")
            print("Download successful\n")
        else:
            print("Download failed\n")
        
        # Test 3: List files
        print("Test 3: Listing files...")
        files = s3.list_files('test/')
        print(f"Files in 'test/': {files}\n")
        
        # Test 4: Check if file exists
        print("Test 4: Checking file existence...")
        exists = s3.file_exists('test/sample.json')
        print(f"File exists: {exists}\n")
        
        # Test 5: Generate presigned URL
        print("Test 5: Generating presigned URL...")
        url = s3.generate_presigned_url('test/sample.json', expiration=300)
        if url:
            print(f"Presigned URL (valid for 5 min): {url[:100]}...\n")
        
        # Test 6: Create folder structure
        print("Test 6: Creating folder structure...")
        s3.create_folder_structure()
        print("Folder structure created\n")
        
        print("="*60)
        print("All tests completed!")
        print("="*60)
        
    except Exception as e:
        print(f"\n Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()
print("hello")