"""
AWS Access Setup Helper
=======================
Interactive script to help set up AWS credentials and Bedrock access.

Author: Your Name
Date: November 2024
"""

import os
import sys
from pathlib import Path

def check_aws_cli():
    """Check if AWS CLI is installed"""
    import subprocess
    try:
        result = subprocess.run(['aws', '--version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print(f"[OK] AWS CLI installed: {result.stdout.strip()}")
            return True
        else:
            print("[ERROR] AWS CLI not found")
            return False
    except FileNotFoundError:
        print("[ERROR] AWS CLI not installed")
        print("\nPlease install AWS CLI:")
        print("  https://aws.amazon.com/cli/")
        return False
    except Exception as e:
        print(f"[ERROR] Error checking AWS CLI: {e}")
        return False

def check_aws_credentials():
    """Check if AWS credentials are configured"""
    try:
        import boto3
        sts = boto3.client('sts')
        identity = sts.get_caller_identity()
        print(f"[OK] AWS credentials configured")
        print(f"  Account ID: {identity.get('Account', 'N/A')}")
        print(f"  User ARN: {identity.get('Arn', 'N/A')}")
        return True
    except Exception as e:
        print(f"[ERROR] AWS credentials not configured: {e}")
        return False

def check_bedrock_access():
    """Check if Bedrock is accessible"""
    try:
        import boto3
        bedrock = boto3.client('bedrock', region_name='us-east-1')
        # Try to list models
        response = bedrock.list_foundation_models()
        print(f"[OK] Bedrock access successful")
        print(f"  Found {len(response.get('modelSummaries', []))} models available")
        return True
    except Exception as e:
        print(f"[ERROR] Bedrock access failed: {e}")
        return False

def check_model_access():
    """Check if specific models are accessible"""
    try:
        import boto3
        bedrock = boto3.client('bedrock', region_name='us-east-1')
        response = bedrock.list_foundation_models()
        models = response.get('modelSummaries', [])
        model_ids = [m['modelId'] for m in models]
        
        # Check for Mixtral
        mixtral_id = "mistral.mixtral-8x7b-instruct-v0:1"
        if any(mixtral_id in mid for mid in model_ids):
            print(f"[OK] Mixtral model accessible: {mixtral_id}")
        else:
            print(f"[ERROR] Mixtral model not accessible: {mixtral_id}")
            print("  Please enable in Bedrock console: https://console.aws.amazon.com/bedrock/")
        
        # Check for Claude
        claude_id = "anthropic.claude-3-5-sonnet-20241022-v2:0"
        if any(claude_id in mid for mid in model_ids):
            print(f"[OK] Claude Sonnet model accessible: {claude_id}")
        else:
            print(f"[ERROR] Claude Sonnet model not accessible: {claude_id}")
            print("  Please enable in Bedrock console: https://console.aws.amazon.com/bedrock/")
        
        return True
    except Exception as e:
        print(f"[ERROR] Error checking model access: {e}")
        return False

def setup_aws_credentials_interactive():
    """Interactive setup for AWS credentials"""
    print("\n" + "=" * 60)
    print("AWS Credentials Setup")
    print("=" * 60)
    print("\nYou need:")
    print("  1. AWS Access Key ID")
    print("  2. AWS Secret Access Key")
    print("  3. Default region (e.g., us-east-1)")
    print("\nTo get your credentials:")
    print("  1. Go to: https://console.aws.amazon.com/iam/")
    print("  2. Click on your user name")
    print("  3. Go to 'Security credentials' tab")
    print("  4. Click 'Create access key'")
    print("  5. Download or copy the credentials")
    print("\n" + "=" * 60)
    
    response = input("\nDo you want to configure AWS credentials now? (y/n): ")
    if response.lower() != 'y':
        print("\nYou can configure later using:")
        print("  aws configure")
        return False
    
    print("\nPlease run the following command to configure:")
    print("  aws configure")
    print("\nOr set environment variables:")
    print("  set AWS_ACCESS_KEY_ID=your_access_key")
    print("  set AWS_SECRET_ACCESS_KEY=your_secret_key")
    print("  set AWS_DEFAULT_REGION=us-east-1")
    
    return False

def setup_bedrock_model_access():
    """Guide for setting up Bedrock model access"""
    print("\n" + "=" * 60)
    print("Bedrock Model Access Setup")
    print("=" * 60)
    print("\nTo enable Bedrock models:")
    print("  1. Go to: https://console.aws.amazon.com/bedrock/")
    print("  2. Click on 'Model access' in the left sidebar")
    print("  3. Click 'Request model access'")
    print("  4. Select the models you need:")
    print("     - Mistral Mixtral 8x7B Instruct")
    print("     - Claude 3.5 Sonnet")
    print("     - Claude 3 Haiku")
    print("  5. Submit the request")
    print("  6. Wait for approval (usually instant for most models)")
    print("\n" + "=" * 60)

def main():
    """Main setup function"""
    print("=" * 60)
    print(" AWS Bedrock Access Setup Helper ".center(60))
    print("=" * 60)
    
    # Check AWS CLI
    print("\n[1] Checking AWS CLI installation...")
    aws_cli_ok = check_aws_cli()
    
    if not aws_cli_ok:
        print("\nPlease install AWS CLI first:")
        print("  https://aws.amazon.com/cli/")
        return
    
    # Check AWS credentials
    print("\n[2] Checking AWS credentials...")
    credentials_ok = check_aws_credentials()
    
    if not credentials_ok:
        setup_aws_credentials_interactive()
        print("\nAfter configuring, run this script again to verify.")
        return
    
    # Check Bedrock access
    print("\n[3] Checking Bedrock access...")
    bedrock_ok = check_bedrock_access()
    
    if not bedrock_ok:
        print("\nBedrock access may require:")
        print("  1. Enabling Bedrock in your AWS account")
        print("  2. Appropriate IAM permissions")
        print("  3. Model access approval")
    
    # Check model access
    print("\n[4] Checking model access...")
    model_ok = check_model_access()
    
    # Summary
    print("\n" + "=" * 60)
    print("Setup Summary")
    print("=" * 60)
    
    all_ok = aws_cli_ok and credentials_ok and bedrock_ok and model_ok
    
    if all_ok:
        print("\n[OK] All checks passed! You're ready to use Bedrock.")
        print("\nYou can now run:")
        print("  python bedrock_music_classifier.py")
    else:
        print("\n[WARNING] Some checks failed. Please fix the issues above.")
        print("\nNext steps:")
        if not credentials_ok:
            print("  1. Configure AWS credentials: aws configure")
        if not bedrock_ok or not model_ok:
            print("  2. Enable Bedrock models: https://console.aws.amazon.com/bedrock/")
    
    print("=" * 60)

if __name__ == "__main__":
    main()

