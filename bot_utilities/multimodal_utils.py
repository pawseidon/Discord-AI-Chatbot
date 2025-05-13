import os
import io
import base64
import aiohttp
import asyncio
import random
import time
import tempfile
from typing import List, Optional, Dict, Any, Union, Tuple
import discord
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from PIL import Image
from bot_utilities.config_loader import config

class ImageProcessor:
    """Module for processing and analyzing images"""
    
    def __init__(self, api_key: str = None, model_name: str = "meta-llama/llama-4-scout-17b-16e-instruct"):
        """Initialize the image processor"""
        self.api_key = api_key or os.environ.get("API_KEY")
        self.model_name = model_name
        self.llm = ChatGroq(
            api_key=self.api_key,
            model_name=self.model_name
        )
    
    async def download_image(self, image_url: str) -> Optional[bytes]:
        """Download image from URL with special handling for Discord's image proxying behavior"""
        try:
            # Use robust headers to handle Discord proxying/caching issues
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.45 Safari/537.36',
                'Accept': 'image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8',
                'Accept-Encoding': 'gzip, deflate, br',
                'Referer': 'https://discord.com/',
                'Cache-Control': 'no-cache',
                'Pragma': 'no-cache'
            }
            
            # Add timestamp parameter to bypass caching
            if '?' not in image_url:
                image_url += f'?t={int(time.time())}'
            else:
                image_url += f'&t={int(time.time())}'
            
            # Detect if URL is from Discord CDN and prepare for proper handling
            is_discord_cdn = 'cdn.discordapp.com' in image_url or 'media.discordapp.net' in image_url
            
            async with aiohttp.ClientSession() as session:
                # First request with extended timeout for Discord CDN
                timeout = aiohttp.ClientTimeout(total=15 if is_discord_cdn else 10)
                async with session.get(image_url, headers=headers, timeout=timeout) as response:
                    if response.status == 200:
                        return await response.read()
                    
                    # If initial request fails and it's a Discord URL, try alternate approach
                    if is_discord_cdn and response.status != 200:
                        print(f"First Discord CDN request failed with status {response.status}, trying alternate approach")
                        
                        # Short delay to avoid rate limiting
                        await asyncio.sleep(1)
                        
                        # Try again with different headers
                        alt_headers = headers.copy()
                        alt_headers['User-Agent'] = 'Discord-AI-Chatbot/1.0'
                        async with session.get(image_url, headers=alt_headers, timeout=timeout) as alt_response:
                            if alt_response.status == 200:
                                return await alt_response.read()
                    
                    print(f"Failed to download image: {response.status}")
                    return None
        except asyncio.TimeoutError:
            print(f"Timeout error downloading image: {image_url}")
            return None
        except Exception as e:
            print(f"Error downloading image: {e}")
            return None
    
    def encode_image_to_base64(self, image_data: bytes) -> str:
        """Encode image data to base64 string"""
        return base64.b64encode(image_data).decode('utf-8')
    
    async def analyze_image(self, image_url: str) -> str:
        """
        Analyze an image and return a description.
        
        Args:
            image_url (str): URL of the image to analyze
            
        Returns:
            str: Description of the image
        """
        try:
            # This is a simplified version - in production you would use an AI vision model
            # For example OpenAI's GPT-4 Vision or another vision model
            
            # Get API info from config
            api_key = os.getenv("OPENAI_API_KEY") or config.get("OPENAI_API_KEY", "")
            
            if not api_key:
                return "Error: OpenAI API key not configured. Please set OPENAI_API_KEY in .env or config.yml."
            
            # Create the API request
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
            
            # Prepare the request with the image URL
            payload = {
                "model": "gpt-4-vision-preview",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe this image in detail. Include all significant visible elements."},
                            {"type": "image_url", "image_url": {"url": image_url}}
                        ]
                    }
                ],
                "max_tokens": 300
            }
            
            # Make the API call
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers=headers,
                    json=payload
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result["choices"][0]["message"]["content"]
                    else:
                        error_text = await response.text()
                        return f"Error processing image: {response.status} - {error_text[:100]}..."
                        
        except Exception as e:
            return f"Error analyzing image: {str(e)}"
    
    async def process_discord_attachment(self, attachment: discord.Attachment, prompt: str = "Describe this image in detail.") -> str:
        """Process a Discord attachment"""
        # Check if the attachment is an image
        if not attachment.content_type or not attachment.content_type.startswith('image/'):
            return "This doesn't appear to be an image attachment."
        
        # Download the image
        image_data = await self.download_image(attachment.url)
        
        if not image_data:
            return "I couldn't download this image."
        
        # Analyze the image
        return await self.analyze_image(attachment.url)
    
    async def extract_text_from_image(self, image_url: str) -> str:
        """
        Extract text from an image (OCR).
        
        Args:
            image_url (str): URL of the image to extract text from
            
        Returns:
            str: Extracted text
        """
        try:
            # This is a simplified version - in production you would use an OCR model
            # For example OpenAI's GPT-4 Vision or another OCR service
            
            # Get API info from config
            api_key = os.getenv("OPENAI_API_KEY") or config.get("OPENAI_API_KEY", "")
            
            if not api_key:
                return "Error: OpenAI API key not configured. Please set OPENAI_API_KEY in .env or config.yml."
            
            # Create the API request
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
            
            # Prepare the request with the image URL
            payload = {
                "model": "gpt-4-vision-preview",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Extract all text from this image, preserving the formatting as much as possible. Only extract the text, don't describe the image."},
                            {"type": "image_url", "image_url": {"url": image_url}}
                        ]
                    }
                ],
                "max_tokens": 500
            }
            
            # Make the API call
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers=headers,
                    json=payload
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result["choices"][0]["message"]["content"]
                    else:
                        error_text = await response.text()
                        return f"Error extracting text: {response.status} - {error_text[:100]}..."
        
        except Exception as e:
            return f"Error performing OCR: {str(e)}"
    
    async def transcribe_audio(self, audio_url: str) -> str:
        """
        Transcribe speech from an audio file to text.
        
        Args:
            audio_url (str): URL of the audio file to transcribe
            
        Returns:
            str: Transcribed text
        """
        try:
            # Get API info from config
            api_key = os.getenv("OPENAI_API_KEY") or config.get("OPENAI_API_KEY", "")
            
            if not api_key:
                return "Error: OpenAI API key not configured. Please set OPENAI_API_KEY in .env or config.yml."
            
            # Download the audio file
            async with aiohttp.ClientSession() as session:
                async with session.get(audio_url) as response:
                    if response.status != 200:
                        return f"Error downloading audio: HTTP {response.status}"
                    
                    # Create a temporary file to store the audio
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
                        temp_file.write(await response.read())
                        temp_file_path = temp_file.name
            
            # Prepare API request headers
            headers = {
                "Authorization": f"Bearer {api_key}"
            }
            
            # Create form data with the audio file
            with open(temp_file_path, "rb") as audio_file:
                form_data = aiohttp.FormData()
                form_data.add_field(
                    name="file",
                    value=audio_file,
                    filename="audio.mp3",
                    content_type="audio/mpeg"
                )
                form_data.add_field("model", "whisper-1")
                form_data.add_field("response_format", "text")
                
                # Make the API call to Whisper
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        "https://api.openai.com/v1/audio/transcriptions",
                        headers=headers,
                        data=form_data
                    ) as response:
                        if response.status == 200:
                            # Simple text response
                            transcript = await response.text()
                            return transcript
                        else:
                            error_text = await response.text()
                            return f"Error transcribing audio: {response.status} - {error_text[:100]}..."
            
        except Exception as e:
            return f"Error transcribing audio: {str(e)}"
        finally:
            # Clean up the temporary file
            if 'temp_file_path' in locals():
                try:
                    os.unlink(temp_file_path)
                except Exception:
                    pass

    async def process_image(self, url: str, operation: str = "analyze") -> str:
        """
        Process an image with the specified operation.
        
        Args:
            url (str): URL of the image to process
            operation (str): Operation to perform (analyze, ocr)
            
        Returns:
            str: Result of the processing
        """
        if operation == "analyze":
            return await self.analyze_image(url)
        elif operation == "ocr":
            return await self.extract_text_from_image(url)
        else:
            return f"Unknown operation: {operation}"

class ImageGenerator:
    """Enhanced module for generating images with multiple providers"""
    
    def __init__(self):
        """Initialize the image generator with defaults"""
        # Default configuration
        self.default_negative_prompt = """
        (deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, 
        extra limb, missing limb, floating limbs, (mutated hands and fingers:1.4), 
        disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation, 
        low quality, lowest quality
        """
        self.timeout = 120  # Timeout in seconds for image generation
        self.retries = 3    # Number of retries on failure
    
    async def generate_with_prodia(self, 
                               prompt: str, 
                               model: str = "dreamshaper-8",
                               negative_prompt: Optional[str] = None, 
                               seed: Optional[int] = None,
                               sampler: str = "DPM++ 2M Karras") -> Tuple[bool, Union[io.BytesIO, str]]:
        """
        Generate image using Prodia API with improved error handling
        
        Returns:
            Tuple[bool, Union[io.BytesIO, str]]: (success, result)
            - If successful, returns (True, BytesIO object with the image)
            - If failed, returns (False, error message)
        """
        start_time = time.time()
        print(f"🖼️ Generating image with Prodia: {prompt}")

        # Setup params
        if seed is None:
            seed = random.randint(1, 9999999)
            
        if negative_prompt is None:
            negative_prompt = self.default_negative_prompt
            
        # Create API request params
        url = 'https://api.prodia.com/generate'
        params = {
            'new': 'true',
            'prompt': prompt,
            'model': model,
            'negative_prompt': negative_prompt,
            'steps': '30',
            'cfg': '7.5',
            'seed': f'{seed}',
            'sampler': sampler,
            'upscale': 'True',
            'aspect_ratio': 'square'
        }
        
        # Try multiple times in case of transient failures
        for attempt in range(self.retries):
            try:
                async with aiohttp.ClientSession() as session:
                    # Step 1: Create job
                    async with session.get(url, params=params, timeout=30) as response:
                        if response.status != 200:
                            error_text = await response.text()
                            continue  # Try again
                        
                        job_data = await response.json()
                        job_id = job_data.get('job')
                        
                        if not job_id:
                            continue  # Try again
                            
                    # Step 2: Poll job until complete
                    job_url = f'https://api.prodia.com/job/{job_id}'
                    headers = {'authority': 'api.prodia.com', 'accept': '*/*'}
                    
                    poll_attempts = 0
                    max_polls = 30  # Maximum number of polling attempts
                    
                    while poll_attempts < max_polls:
                        poll_attempts += 1
                        
                        async with session.get(job_url, headers=headers) as job_response:
                            if job_response.status != 200:
                                await asyncio.sleep(1)
                                continue
                                
                            job_status = await job_response.json()
                            
                            if job_status.get('status') == 'succeeded':
                                # Step 3: Download the image
                                image_url = f'https://images.prodia.xyz/{job_id}.png?download=1'
                                async with session.get(image_url, headers=headers) as img_response:
                                    if img_response.status == 200:
                                        content = await img_response.content.read()
                                        img_file_obj = io.BytesIO(content)
                                        duration = time.time() - start_time
                                        print(f"✅ Generated image in {duration:.2f} seconds")
                                        return True, img_file_obj
                            
                            elif job_status.get('status') == 'failed':
                                break  # Job failed, try a new job
                                
                            # Wait before polling again
                            await asyncio.sleep(2)
                    
            except asyncio.TimeoutError:
                print(f"⚠️ Timeout during image generation (attempt {attempt+1}/{self.retries})")
                await asyncio.sleep(2)
                continue
                
            except Exception as e:
                print(f"⚠️ Error in image generation (attempt {attempt+1}/{self.retries}): {e}")
                await asyncio.sleep(2)
                continue
        
        # If we got here, all attempts failed
        return False, "Failed to generate image after multiple attempts. Please try again later."
    
    async def generate_with_pollinations(self, prompt: str) -> Tuple[bool, Union[io.BytesIO, str]]:
        """
        Generate image using Pollinations.ai API
        
        Returns:
            Tuple[bool, Union[io.BytesIO, str]]: (success, result)
            - If successful, returns (True, BytesIO object with the image)
            - If failed, returns (False, error message)
        """
        try:
            seed = random.randint(1, 100000)
            image_url = f"https://image.pollinations.ai/prompt/{prompt}?seed={seed}"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(image_url, timeout=30) as response:
                    if response.status != 200:
                        return False, f"Failed to generate image. Status code: {response.status}"
                        
                    image_data = await response.read()
                    return True, io.BytesIO(image_data)
                    
        except Exception as e:
            return False, f"Failed to generate image: {str(e)}"
    
    async def enhance_prompt(self, prompt: str) -> str:
        """
        Enhance a simple prompt with more details for better image generation
        This is useful for converting simple user prompts into detailed, high-quality prompts
        """
        # This could be implemented with a simple template or an LLM call
        # For now, just add some quality boosters to the prompt
        enhancers = [
            "high resolution", "highly detailed", "4K", "beautiful lighting", 
            "professional photography", "masterpiece", "sharp focus"
        ]
        
        # Only enhance if the prompt is short/simple
        if len(prompt.split()) < 15:
            enhanced = f"{prompt}, {', '.join(random.sample(enhancers, 3))}"
            return enhanced
        
        return prompt
    
    async def generate_image(self, 
                           prompt: str, 
                           provider: str = "prodia",
                           model: str = "dreamshaper-8", 
                           negative_prompt: Optional[str] = None,
                           enhance: bool = True) -> Tuple[bool, Union[io.BytesIO, str], Dict[str, Any]]:
        """
        Unified method to generate images from multiple providers with fallback
        
        Args:
            prompt: Description of the image to generate
            provider: The service to use ("prodia" or "pollinations")
            model: Model ID for providers that support multiple models
            negative_prompt: What to exclude from the generation
            enhance: Whether to enhance the prompt automatically
            
        Returns:
            Tuple[bool, Union[io.BytesIO, str], Dict[str, Any]]: 
            - success flag
            - image data or error message
            - metadata dictionary with generation details
        """
        # Start metadata collection
        metadata = {
            "original_prompt": prompt,
            "provider": provider,
            "model": model,
            "generation_time": 0,
            "enhanced": enhance
        }
        
        start_time = time.time()
        
        # Optionally enhance the prompt
        if enhance:
            prompt = await self.enhance_prompt(prompt)
            metadata["enhanced_prompt"] = prompt
        
        # Try primary provider
        primary_success = False
        primary_result = None
        
        if provider.lower() == "prodia":
            primary_success, primary_result = await self.generate_with_prodia(
                prompt=prompt,
                model=model,
                negative_prompt=negative_prompt
            )
        elif provider.lower() == "pollinations":
            primary_success, primary_result = await self.generate_with_pollinations(prompt)
        else:
            # Default to Prodia if unknown provider
            primary_success, primary_result = await self.generate_with_prodia(
                prompt=prompt,
                model=model,
                negative_prompt=negative_prompt
            )
        
        # Calculate generation time
        metadata["generation_time"] = time.time() - start_time
        
        # If primary provider succeeded, return the result
        if primary_success:
            return True, primary_result, metadata
            
        # Try fallback provider if primary failed
        if provider.lower() != "pollinations":
            print("⚠️ Primary provider failed, trying Pollinations.ai as fallback")
            fallback_success, fallback_result = await self.generate_with_pollinations(prompt)
            
            if fallback_success:
                metadata["provider"] = "pollinations (fallback)"
                return True, fallback_result, metadata
        
        # All providers failed
        return False, primary_result, metadata  # primary_result contains the error message

class MultimodalMessage:
    """Utility for creating multimodal messages with text and images"""
    
    @staticmethod
    async def create_multimodal_prompt(text: str, image_urls: List[str] = None) -> List[Dict[str, Any]]:
        """Create a multimodal prompt with text and images"""
        content = [{"type": "text", "text": text}]
        
        if image_urls:
            for url in image_urls:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": url}
                })
        
        return content
    
    @staticmethod
    async def format_discord_response(text: str, image_url: Optional[str] = None) -> Dict[str, Any]:
        """Format a response for Discord with optional image"""
        response = {"content": text}
        
        if image_url:
            # Create an embed with the image
            embed = discord.Embed()
            embed.set_image(url=image_url)
            response["embed"] = embed
        
        return response 