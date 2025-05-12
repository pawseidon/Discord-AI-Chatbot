import asyncio
import os
import sys
import json
import aiohttp
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

# Use direct IP address of Windows host
WINDOWS_IP = "172.29.160.1"
LM_STUDIO_PORT = "1234"
LM_STUDIO_URL = f"http://{WINDOWS_IP}:{LM_STUDIO_PORT}/v1"
MODEL_ID = "mistral-nemo-instruct-2407"  # Updated model

async def check_server_endpoint():
    print(f"Testing basic connectivity to {LM_STUDIO_URL}...")
    try:
        async with aiohttp.ClientSession() as session:
            # Just try to connect to the models endpoint
            async with session.get(f"{LM_STUDIO_URL}/models") as response:
                if response.status == 200:
                    data = await response.json()
                    print("Server connection successful!")
                    print(f"Available models: {json.dumps(data, indent=2)}")
                    return True
                else:
                    print(f"Server responded with status code: {response.status}")
                    body = await response.text()
                    print(f"Response body: {body[:500]}")  # Show first 500 chars if long
                    return False
    except Exception as e:
        print(f"Server connection error: {type(e).__name__}: {e}")
        return False

async def test_local_connection():
    print(f"Testing connection to local LM Studio model at {LM_STUDIO_URL}...")
    
    # First verify server connectivity
    if not await check_server_endpoint():
        print("\nServer connectivity test failed.")
        print("Please check that:")
        print("1. LM Studio is running")
        print("2. Local Server has been started in LM Studio")
        print("3. Server is configured for port 1234")
        print("4. 'Serve on local network' is enabled in LM Studio")
        print("5. CORS is enabled for API access")
        return False
    
    client = AsyncOpenAI(
        base_url=LM_STUDIO_URL,
        api_key="not-needed"  # LM Studio doesn't require an API key
    )
    
    try:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, are you working correctly?"}
        ]
        
        print("Sending test chat completion request...")
        print(f"Using model: {MODEL_ID}")
        response = await client.chat.completions.create(
            model=MODEL_ID,
            messages=messages,
        )
        
        print("Connection successful!")
        print("Response from model:")
        print(response.choices[0].message.content)
        return True
    except Exception as e:
        print(f"Error with chat completion: {type(e).__name__}: {e}")
        print("\nPossible solutions:")
        print(f"1. Make sure the model '{MODEL_ID}' is loaded in LM Studio")
        print("2. Check if the OpenAI API compatibility is enabled in LM Studio settings")
        print("3. Try with a different model name that matches exactly what's loaded in LM Studio")
        return False

if __name__ == "__main__":
    print("LM Studio Connection Test")
    print("-----------------------")
    asyncio.run(test_local_connection()) 