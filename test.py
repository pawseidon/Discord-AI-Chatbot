import os
from openai import Client 
from dotenv import load_dotenv
from bot_utilities.config_loader import config

load_dotenv()

# Get the base URL and ensure it's properly formatted
api_base = config['API_BASE_URL']

# Remove trailing slash if present
if api_base.endswith('/'):
    api_base = api_base[:-1]

# If it ends with /chat/completions, strip that off
if api_base.endswith('/chat/completions'):
    api_base = api_base.rsplit('/chat/completions', 1)[0]

print(f"Connecting to remote API at: {api_base}")

client = Client(
    base_url=api_base,
    api_key=os.environ.get("API_KEY"),
)
models = client.models.list()
for model in models.data:
    if model.active:
        try:
            response = client.chat.completions.create(
                    messages=[
                        {
                            "role": "user",
                            "content": "Say this is a test",
                        }
                    ],
                    model=model.id
                )
            print(f"{model.id} responded with : {response.choices[0].message.content}\n\n")
        except Exception as e:
            print(f'{model.id} failed : {e}\n\n')