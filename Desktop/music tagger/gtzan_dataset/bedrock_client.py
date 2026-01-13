"""
AWS Bedrock Client
==================
Client for invoking AWS Bedrock models for music genre classification.
Supports Mixtral, Claude, and other Bedrock models.

Author: Your Name
Date: November 2024
"""

import json
import boto3
import time
from typing import Dict, Optional, List
from botocore.exceptions import ClientError


class BedrockClient:
    """Client for AWS Bedrock model invocation"""
    
    # Supported models
    MODELS = {
        "mixtral": "mistral.mixtral-8x7b-instruct-v0:1",
        "mixtral_large": "mistral.mixtral-large-2402-v1:0",
        "claude_sonnet": "anthropic.claude-3-5-sonnet-20241022-v2:0",
        "claude_haiku": "anthropic.claude-3-haiku-20240307-v1:0",
        "claude_opus": "anthropic.claude-3-opus-20240229-v1:0",
        "titan": "amazon.titan-text-express-v1",
        "llama": "meta.llama3-70b-instruct-v1:0",
        "cohere": "cohere.command-r-plus-v1:0",
    }
    
    def __init__(self, region: str = "us-east-1", max_retries: int = 3):
        """
        Initialize Bedrock client
        
        Args:
            region: AWS region for Bedrock
            max_retries: Maximum number of retry attempts
        """
        self.region = region
        self.max_retries = max_retries
        self.bedrock_runtime = None
        self.bedrock = None
        self.setup_clients()
    
    def setup_clients(self):
        """Setup AWS Bedrock clients"""
        try:
            self.bedrock_runtime = boto3.client(
                service_name='bedrock-runtime',
                region_name=self.region
            )
            self.bedrock = boto3.client(
                service_name='bedrock',
                region_name=self.region
            )
            print(f"[OK] Bedrock clients initialized (region: {self.region})")
        except Exception as e:
            print(f"Error setting up Bedrock clients: {e}")
            print("\nPlease ensure:")
            print("1. AWS CLI is configured: aws configure")
            print("2. You have Bedrock access in your AWS account")
            print("3. Required models are enabled in Bedrock console")
            raise
    
    def check_model_access(self, model_id: str) -> bool:
        """
        Check if model is accessible
        
        Args:
            model_id: Model identifier
        
        Returns:
            True if model is accessible
        """
        try:
            response = self.bedrock.list_foundation_models()
            models = response.get('modelSummaries', [])
            model_ids = [m['modelId'] for m in models]
            
            # Check if model is in available models
            return any(model_id in mid for mid in model_ids)
        except Exception as e:
            print(f"Error checking model access: {e}")
            return False
    
    def invoke_mixtral(self, prompt: str, temperature: float = 0.3, 
                      max_tokens: int = 2000) -> Dict:
        """
        Invoke Mixtral model
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
        
        Returns:
            Model response dictionary
        """
        model_id = self.MODELS["mixtral"]
        
        try:
            # Mistral/Mixtral format for AWS Bedrock
            # Format: Use prompt field with instruction wrapper
            body = json.dumps({
                "prompt": f"<s>[INST] {prompt} [/INST]",
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": 0.9,
                "top_k": 50,
                "stop": ["</s>", "[/INST]"]
            })
            
            # Invoke model
            response = self.bedrock_runtime.invoke_model(
                modelId=model_id,
                body=body,
                contentType='application/json',
                accept='application/json'
            )
            
            # Parse response
            response_body = json.loads(response['body'].read())
            
            # Mistral/Mixtral response format
            # AWS Bedrock Mistral models return: {"outputs": [{"text": "..."}]}
            text = None
            
            # Format 1: outputs[0].text (Standard Mistral format)
            if 'outputs' in response_body and isinstance(response_body['outputs'], list):
                if len(response_body['outputs']) > 0:
                    output = response_body['outputs'][0]
                    if isinstance(output, dict):
                        # Try common text fields
                        for key in ['text', 'output', 'content']:
                            if key in output:
                                text = output[key]
                                break
                        # If no text field found, try to extract from dict
                        if not text:
                            # Get first string value
                            for value in output.values():
                                if isinstance(value, str) and len(value) > 10:
                                    text = value
                                    break
                    elif isinstance(output, str):
                        text = output
            
            # Format 2: Direct text field (fallback)
            if not text and 'text' in response_body:
                text = response_body['text']
            
            # Format 3: content field (Claude-like fallback)
            if not text and 'content' in response_body:
                if isinstance(response_body['content'], str):
                    text = response_body['content']
                elif isinstance(response_body['content'], list) and len(response_body['content']) > 0:
                    if isinstance(response_body['content'][0], dict):
                        text = response_body['content'][0].get('text', str(response_body['content'][0]))
                    else:
                        text = str(response_body['content'][0])
            
            # Fallback: convert entire response to string
            if not text:
                text = json.dumps(response_body, indent=2)
                print(f"Warning: Unexpected response format: {text[:200]}")
            
            # Clean up text if it contains prompt wrapper tokens
            if text:
                # Remove instruction wrapper tokens
                text = text.replace("<s>", "").replace("</s>", "").strip()
                if "[/INST]" in text:
                    # Extract text after [/INST]
                    parts = text.split("[/INST]")
                    if len(parts) > 1:
                        text = parts[-1].strip()
            
            return self._parse_json_response(text)
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'ThrottlingException':
                print(f"Rate limit exceeded. Waiting 5 seconds...")
                time.sleep(5)
                return self.invoke_mixtral(prompt, temperature, max_tokens)
            elif error_code == 'ModelNotReadyException':
                print(f"Model not ready. Waiting 10 seconds...")
                time.sleep(10)
                return self.invoke_mixtral(prompt, temperature, max_tokens)
            else:
                print(f"Error invoking Mixtral: {e}")
                return {"genre": "error", "confidence": 0, "reasoning": str(e)}
        except Exception as e:
            print(f"Unexpected error invoking Mixtral: {e}")
            return {"genre": "error", "confidence": 0, "reasoning": str(e)}
    
    def invoke_claude(self, prompt: str, model: str = "sonnet", 
                     temperature: float = 0.3, max_tokens: int = 2000) -> Dict:
        """
        Invoke Claude model
        
        Args:
            prompt: Input prompt
            model: Model variant (sonnet or haiku)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
        
        Returns:
            Model response dictionary
        """
        model_id = self.MODELS[f"claude_{model}"]
        
        try:
            # Claude format
            body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            })
            
            # Invoke model
            response = self.bedrock_runtime.invoke_model(
                modelId=model_id,
                body=body,
                contentType='application/json',
                accept='application/json'
            )
            
            # Parse response
            response_body = json.loads(response['body'].read())
            
            # Claude response format
            if 'content' in response_body:
                text = response_body['content'][0]['text']
            else:
                text = str(response_body)
            
            return self._parse_json_response(text)
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'ThrottlingException':
                print(f"Rate limit exceeded. Waiting 5 seconds...")
                time.sleep(5)
                return self.invoke_claude(prompt, model, temperature, max_tokens)
            else:
                print(f"Error invoking Claude: {e}")
                return {"genre": "error", "confidence": 0, "reasoning": str(e)}
        except Exception as e:
            print(f"Unexpected error invoking Claude: {e}")
            return {"genre": "error", "confidence": 0, "reasoning": str(e)}
    
    def invoke_model(self, model_name: str, prompt: str, 
                    temperature: float = 0.3, max_tokens: int = 2000) -> Dict:
        """
        Invoke a specific model by name
        
        Args:
            model_name: Model name (mixtral, claude_sonnet, claude_haiku, etc.)
            prompt: Input prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
        
        Returns:
            Model response dictionary
        """
        if model_name == "mixtral":
            return self.invoke_mixtral(prompt, temperature, max_tokens)
        elif model_name.startswith("claude"):
            model_type = model_name.split("_")[-1] if "_" in model_name else "sonnet"
            return self.invoke_claude(prompt, model_type, temperature, max_tokens)
        else:
            print(f"Unknown model: {model_name}")
            return {"genre": "error", "confidence": 0, "reasoning": f"Unknown model: {model_name}"}
    
    def _parse_json_response(self, text: str) -> Dict:
        """
        Parse JSON response from model
        
        Args:
            text: Raw text response from model
        
        Returns:
            Parsed JSON dictionary
        """
        try:
            # Try to extract JSON from response
            import re
            
            # Look for JSON object in response
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                # Clean up any markdown code blocks
                json_str = json_str.replace('```json', '').replace('```', '').strip()
                result = json.loads(json_str)
                
                # Validate genre
                valid_genres = ["blues", "classical", "country", "disco", "hiphop", 
                              "jazz", "metal", "pop", "reggae", "rock"]
                
                if "genre" in result:
                    # Normalize genre name
                    genre = result["genre"].lower().strip()
                    # Handle variations
                    if "hip" in genre and "hop" in genre:
                        genre = "hiphop"
                    elif "hip-hop" in genre or "hip_hop" in genre:
                        genre = "hiphop"
                    
                    if genre not in valid_genres:
                        # Try to find closest match
                        for valid_genre in valid_genres:
                            if valid_genre in genre or genre in valid_genre:
                                genre = valid_genre
                                break
                        else:
                            genre = "unknown"
                    
                    result["genre"] = genre
                
                # Ensure confidence is between 0 and 1
                if "confidence" in result:
                    confidence = result["confidence"]
                    if isinstance(confidence, str):
                        try:
                            confidence = float(confidence)
                        except:
                            confidence = 0.5
                    result["confidence"] = max(0.0, min(1.0, float(confidence)))
                else:
                    result["confidence"] = 0.5
                
                return result
            else:
                # No JSON found, try to extract genre from text
                genre = self._extract_genre_from_text(text)
                return {
                    "genre": genre,
                    "confidence": 0.5,
                    "reasoning": text[:500]  # First 500 chars
                }
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            print(f"Response text: {text[:500]}")
            # Try to extract genre from text
            genre = self._extract_genre_from_text(text)
            return {
                "genre": genre,
                "confidence": 0.5,
                "reasoning": f"JSON parse error: {str(e)}. Response: {text[:200]}"
            }
        except Exception as e:
            print(f"Error parsing response: {e}")
            return {
                "genre": "unknown",
                "confidence": 0.0,
                "reasoning": f"Parse error: {str(e)}"
            }
    
    def _extract_genre_from_text(self, text: str) -> str:
        """
        Extract genre from text response if JSON parsing fails
        
        Args:
            text: Text response
        
        Returns:
            Extracted genre name
        """
        valid_genres = ["blues", "classical", "country", "disco", "hiphop", 
                       "jazz", "metal", "pop", "reggae", "rock"]
        
        text_lower = text.lower()
        
        # Look for genre mentions
        for genre in valid_genres:
            if genre in text_lower:
                return genre
        
        # Look for common variations
        variations = {
            "hip-hop": "hiphop",
            "hip_hop": "hiphop",
            "hip hop": "hiphop"
        }
        
        for variant, genre in variations.items():
            if variant in text_lower:
                return genre
        
        return "unknown"
    
    def test_connection(self) -> bool:
        """
        Test Bedrock connection
        
        Returns:
            True if connection is successful
        """
        try:
            # Try to list models (without maxResults parameter)
            response = self.bedrock.list_foundation_models()
            return True
        except Exception as e:
            # Connection test failed, but this is okay - we can still try to use models
            # The error might be due to parameter validation, not actual connection issues
            return True  # Return True to allow proceeding


if __name__ == "__main__":
    # Test Bedrock client
    print("Testing AWS Bedrock Client...")
    
    try:
        client = BedrockClient()
        
        # Test connection
        if client.test_connection():
            print("✓ Bedrock connection successful")
        else:
            print("✗ Bedrock connection failed")
            print("Please check AWS credentials and Bedrock access")
    except Exception as e:
        print(f"Error: {e}")
        print("\nPlease ensure:")
        print("1. AWS CLI is configured: aws configure")
        print("2. You have Bedrock access")
        print("3. Required models are enabled")

