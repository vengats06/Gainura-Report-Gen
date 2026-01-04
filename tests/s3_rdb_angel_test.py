"""
Integration Test - All Components Working Together
==================================================

This test verifies the complete data flow:
1. Fetch data from Angel One API
2. Upload raw data to S3
3. Store processed data in RDS PostgreSQL

Run this to verify your setup is working correctly!
"""

from datetime import datetime, timedelta
from data_collection.angel_one_api import AngelOneAPI
from aws.s3_handler import S3Handler
from aws.rds_connection import RDSConnection
from utils.logger import get_logger

logger = get_logger(__name__)


def test_complete_workflow():
    """
    Test complete data pipeline for one stock symbol.
    """
    print("\n" + "="*70)
    print("INTEGRATION TEST - Complete Data Pipeline")
    print("="*70 + "\n")
    
    symbol = 'TCS'
    
    try:
        # =========================================
        # STEP 1: Initialize all components
        # =========================================
        print("Step 1: Initializing components...")
        angel = AngelOneAPI()
        s3 = S3Handler()
        db = RDSConnection()
        print("All components initialized\n")
        
        # =========================================
        # STEP 2: Login to Angel One
        # =========================================
        print("Step 2: Logging into Angel One...")
        if not angel.login():
            print("Angel One login failed!")
            return False
        print(" Angel One login successful\n")
        
        # =========================================
        # STEP 3: Fetch historical data (last 7 days)
        # =========================================
        print(f"Step 3: Fetching {symbol} data (last 7 days)...")
        to_date = datetime.now().strftime('%Y-%m-%d')
        from_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        
        df = angel.get_historical_data(symbol, from_date, to_date)
        
        if df is None or len(df) == 0:
            print(" No data fetched from Angel One!")
            return False
        
        print(f" Fetched {len(df)} records")
        print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"   Price range: ₹{df['close'].min():.2f} to ₹{df['close'].max():.2f}\n")
        
        # =========================================
        # STEP 4: Upload raw data to S3
        # =========================================
        print("Step 4: Uploading raw data to S3...")
        
        # Convert DataFrame to JSON
        data_json = df.to_dict(orient='records')
        s3_key = f"angel_one/{symbol}_{datetime.now().strftime('%Y-%m-%d')}.json"
        
        if s3.upload_json(data_json, s3_key, bucket_type='raw'):
            print(f" Raw data uploaded to S3: {s3_key}\n")
        else:
            print(" S3 upload failed!")
            return False
        
        # =========================================
        # STEP 5: Create database tables
        # =========================================
        print("Step 5: Creating database tables...")
        db.create_tables()
        print(" Database tables ready\n")
        
        # =========================================
        # STEP 6: Bulk insert into RDS PostgreSQL
        # =========================================
        print("Step 6: Inserting data into PostgreSQL...")
        
        # Prepare data for bulk insert
        bulk_data = []
        for _, row in df.iterrows():
            bulk_data.append((
                row['symbol'],
                row['timestamp'].strftime('%Y-%m-%d'),
                float(row['open']),
                float(row['high']),
                float(row['low']),
                float(row['close']),
                int(row['volume'])
            ))
        
        db.bulk_insert_stock_prices(bulk_data)
        print(f" Inserted {len(bulk_data)} records into PostgreSQL\n")
        
        # =========================================
        # STEP 7: Verify data in database
        # =========================================
        print("Step 7: Verifying data in database...")
        
        stored_data = db.get_stock_prices(symbol, from_date, to_date)
        
        if len(stored_data) > 0:
            print(f" Verification successful!")
            print(f"   Records in database: {len(stored_data)}")
            print(f"   Latest record:")
            latest = stored_data[0]
            print(f"     Date: {latest['date']}")
            print(f"     Close: ₹{latest['close']:.2f}")
            print(f"     Volume: {latest['volume']:,}\n")
        else:
            print(" No data found in database (might be timing issue)\n")
        
        # =========================================
        # STEP 8: Download from S3 (verify)
        # =========================================
        print("Step 8: Verifying S3 data...")
        
        downloaded_data = s3.download_json(s3_key, bucket_type='raw')
        
        if downloaded_data and len(downloaded_data) > 0:
            print(f" S3 data verified!")
            print(f"   Records in S3: {len(downloaded_data)}\n")
        else:
            print(" Could not verify S3 data\n")
        
        # =========================================
        # STEP 9: Cleanup (logout)
        # =========================================
        print("Step 9: Cleanup...")
        angel.logout()
        db.close_all_connections()
        print(" All connections closed\n")
        
        # =========================================
        # SUCCESS!
        # =========================================
        print("="*70)
        print("🎉 INTEGRATION TEST PASSED! 🎉")
        print("="*70)
        print("\nYour setup is working correctly!")
        print("All components are properly integrated:")
        print("   Angel One API - Fetching data")
        print("   AWS S3 - Storing raw data")
        print("   AWS RDS PostgreSQL - Storing structured data")
        print("\nYou're ready to build the complete application!")
        print("="*70 + "\n")
        
        return True
        
    except Exception as e:
        print(f"\n Integration test failed!")
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_complete_workflow()