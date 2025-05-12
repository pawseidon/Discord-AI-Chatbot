# Setting Up a Local LLM with LM Studio

This guide explains how to use the Discord AI Chatbot with a local language model running through LM Studio.

## Prerequisites

1. Install [LM Studio](https://lmstudio.ai/) on your computer
2. Make sure you have sufficient disk space for the model (typically 4-10GB)
3. A computer with enough RAM (at least 8GB, recommended 16GB+) and preferably a GPU

## Setup Steps

### 1. Download and Set Up LM Studio

1. Download and install LM Studio from [lmstudio.ai](https://lmstudio.ai/)
2. Open LM Studio
3. Go to the "Models" tab
4. Search for "deepseek-r1-distill-llama-8b" and download it
   - Alternatively, you can use any other compatible model

### 2. Start the Local Server in LM Studio

1. Select the downloaded model from your local models list
2. Click on "Chat" to load the model
3. Go to the "Local Server" tab at the bottom of the window
4. Click "Start Server"
5. Make sure the server is running on `http://127.0.0.1:1234`
6. Verify that "OpenAI compatible server" is enabled

### 3. Configure the Discord Bot

The `config.yml` file is already configured to use the local model with the following settings:

```yaml
USE_LOCAL_MODEL: true
LOCAL_MODEL_URL: http://127.0.0.1:1234/v1
LOCAL_MODEL_ID: deepseek-r1-distill-llama-8b
```

If you want to use a different model, update the `LOCAL_MODEL_ID` with the name of your chosen model.

### 4. Testing the Connection

Before running the main bot, you can test the connection to your local model:

```bash
python test_local_model.py
```

This will attempt to connect to your local LM Studio server and generate a response. If successful, you'll see the model's response in the terminal.

### 5. Run the Discord Bot

Once the test is successful, you can run the main bot:

```bash
python main.py
```

## Troubleshooting

### Common Issues

1. **Connection Error**:
   - Make sure LM Studio is running and the server is started
   - Verify the server URL is correct in your config.yml
   - Check that you have the right model loaded in LM Studio

2. **Slow Responses**:
   - Local models may be slower than cloud-based ones, especially on computers without a GPU
   - Consider reducing the context length or using a smaller model

3. **Out of Memory Errors**:
   - Try using a smaller model
   - Close other applications to free up RAM
   - Adjust the model settings in LM Studio (lower context length)

4. **Model Not Found**:
   - Make sure the model name in `config.yml` exactly matches the one in LM Studio

## Switching Between Local and Cloud Models

If you want to switch back to using a cloud-based model (like Groq's API), simply change the following in your `config.yml`:

```yaml
USE_LOCAL_MODEL: false
```

This will revert to using the remote API specified in the `API_BASE_URL` with your API key from the `.env` file.

## Advanced Configuration

- To change the local server port or address, update the `LOCAL_MODEL_URL` in `config.yml`
- You can experiment with different local models by changing the `LOCAL_MODEL_ID`
- For more advanced settings, refer to the LM Studio documentation 