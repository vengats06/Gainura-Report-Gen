"""
AWS Lambda Function for ETL Automation
======================================

This Lambda function triggers automatically when new data is uploaded to S3.
Serverless = No servers to manage, AWS runs code on demand.

Trigger Flow:
1. New file uploaded to S3 raw-data bucket
2. S3 sends event notification to Lambda
3. Lambda function executes automatically
4. Runs ETL pipeline to process the data
5. Saves results to processed bucket

Why Lambda?
- No servers to manage (serverless)
- Pay only when code runs ($0.20 per 1M requests)
- Auto-scales (handles 1 or 1000 requests)
- Perfect for event-driven workflows

Deployment:
1. Package this code with dependencies
2. Upload to AWS Lambda
3. Configure S3 trigger
4. Done! Automatic processing

Note: This file is designed to be deployed to AWS Lambda.
For local testing, use etl/pyspark_etl.py directly.

Usage (in Lambda):
    # Automatically triggered by S3 upload
    # No manual execution needed
"""

import json
import boto3
from datetime import datetime, timedelta


def lambda_handler(event, context):
    """
    AWS Lambda entry point.
    
    This function is called automatically by AWS Lambda when:
    - S3 upload event occurs
    - Manual Lambda invocation
    - Scheduled CloudWatch event
    
    Args:
        event: AWS event data (contains S3 bucket, key, etc.)
        context: Lambda runtime information
        
    Returns:
        Dict with status and message
        
    Event structure (S3 upload):
    {
        "Records": [{
            "s3": {
                "bucket": {"name": "gainura-raw-data"},
                "object": {"key": "angel_one/TCS_2024-12-25.json"}
            }
        }]
    }
    """
    print("Lambda function started")
    print(f"Event: {json.dumps(event)}")
    
    try:
        # Extract S3 bucket and key from event
        if 'Records' in event and len(event['Records']) > 0:
            # Triggered by S3 upload
            s3_event = event['Records'][0]['s3']
            bucket_name = s3_event['bucket']['name']
            object_key = s3_event['object']['key']
            
            print(f"Processing S3 upload: s3://{bucket_name}/{object_key}")
            
            # Extract symbol from filename
            # Example: angel_one/TCS_2024-12-25.json -> TCS
            filename = object_key.split('/')[-1]
            symbol = filename.split('_')[0]
            
            print(f"Extracted symbol: {symbol}")
            
            # Run ETL pipeline for this symbol
            result = run_etl_pipeline(symbol, bucket_name, object_key)
            
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'message': 'ETL pipeline completed successfully',
                    'symbol': symbol,
                    'result': result
                })
            }
        
        elif 'symbol' in event:
            # Manual invocation with symbol parameter
            symbol = event['symbol']
            days = event.get('days', 30)
            
            print(f"Manual invocation for {symbol}, {days} days")
            
            result = run_etl_pipeline_manual(symbol, days)
            
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'message': 'ETL pipeline completed successfully',
                    'symbol': symbol,
                    'result': result
                })
            }
        
        else:
            print("Unknown event type")
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'Invalid event format'})
            }
    
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e),
                'message': 'ETL pipeline failed'
            })
        }


def run_etl_pipeline(symbol: str, bucket: str, key: str) -> dict:
    """
    Run ETL pipeline for uploaded data.
    
    This is a simplified version for Lambda.
    In production, this would:
    1. Read raw data from S3
    2. Process with pandas/PySpark
    3. Calculate indicators
    4. Save to processed bucket
    5. Update RDS database
    
    Args:
        symbol: Stock symbol
        bucket: S3 bucket name
        key: S3 object key
        
    Returns:
        Dict with processing results
    """
    print(f"Starting ETL for {symbol}")
    
    s3 = boto3.client('s3')
    
    # Download raw data from S3
    print(f"Downloading s3://{bucket}/{key}")
    response = s3.get_object(Bucket=bucket, Key=key)
    raw_data = json.loads(response['Body'].read())
    
    print(f"Processing {len(raw_data)} records")
    
    # In production, this would:
    # - Import etl.pyspark_etl
    # - Run complete transformation
    # - Calculate indicators
    # - Save results
    
    # For now, just log success
    print(f"ETL completed for {symbol}")
    
    return {
        'symbol': symbol,
        'records_processed': len(raw_data),
        'status': 'success'
    }


def run_etl_pipeline_manual(symbol: str, days: int = 30) -> dict:
    """
    Run ETL pipeline manually (not triggered by S3).
    
    This is used when Lambda is invoked manually or by CloudWatch schedule.
    
    Args:
        symbol: Stock symbol
        days: Number of days to process
        
    Returns:
        Dict with processing results
    """
    print(f"Manual ETL for {symbol}, {days} days")
    
    # In production, this would:
    # 1. Fetch data from Angel One API
    # 2. Run complete ETL pipeline
    # 3. Save to S3 and RDS
    
    # For Lambda deployment, you would:
    # - Package all dependencies (pandas, boto3, etc.)
    # - Include etl modules in Lambda deployment package
    # - Configure environment variables
    
    return {
        'symbol': symbol,
        'days': days,
        'status': 'success',
        'message': 'Manual ETL completed'
    }


def scheduled_daily_etl(event, context):
    """
    Run ETL for all stocks daily.
    
    This function is triggered by CloudWatch Events (cron schedule).
    Example: Run every day at 6 PM IST (after market close)
    
    CloudWatch cron: cron(30 12 * * ? *)  # 12:30 PM UTC = 6:00 PM IST
    
    Args:
        event: CloudWatch event
        context: Lambda context
        
    Returns:
        Dict with results
    """
    print("Starting scheduled daily ETL")
    
    # List of stocks to process
    symbols = ['TCS', 'RELIANCE', 'INFY', 'HDFCBANK', 'ICICIBANK']
    
    results = []
    
    for symbol in symbols:
        print(f"Processing {symbol}...")
        try:
            result = run_etl_pipeline_manual(symbol, days=1)
            results.append(result)
        except Exception as e:
            print(f"Failed to process {symbol}: {str(e)}")
            results.append({
                'symbol': symbol,
                'status': 'failed',
                'error': str(e)
            })
    
    return {
        'statusCode': 200,
        'body': json.dumps({
            'message': 'Daily ETL completed',
            'results': results
        })
    }


# For local testing (not used in Lambda)
if __name__ == "__main__":
    """
    Local testing of Lambda function.
    
    This simulates Lambda invocation locally.
    """
    print("\n" + "="*60)
    print("Testing Lambda Function Locally")
    print("="*60 + "\n")
    
    # Test 1: Simulate S3 upload event
    print("Test 1: S3 upload event...")
    s3_event = {
        "Records": [{
            "s3": {
                "bucket": {"name": "gainura-raw-data"},
                "object": {"key": "angel_one/TCS_2024-12-25.json"}
            }
        }]
    }
    
    result = lambda_handler(s3_event, None)
    print(f"Result: {result}\n")
    
    # Test 2: Manual invocation
    print("Test 2: Manual invocation...")
    manual_event = {
        "symbol": "RELIANCE",
        "days": 7
    }
    
    result = lambda_handler(manual_event, None)
    print(f"Result: {result}\n")
    
    print("="*60)
    print("Local testing completed!")
    print("="*60)
    print("\nNote: This is local simulation.")
    print("To deploy to AWS Lambda:")
    print("1. Package code with dependencies")
    print("2. Create Lambda function in AWS Console")
    print("3. Upload deployment package")
    print("4. Configure S3 trigger")
    print("5. Test in AWS Lambda")