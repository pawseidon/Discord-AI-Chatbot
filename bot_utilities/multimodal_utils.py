import os
import io
import base64
import aiohttp
import asyncio
import random
import time
from typing import List, Optional, Dict, Any, Union, Tuple
import discord
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

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
        """Download image from URL"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(image_url) as response:
                    if response.status == 200:
                        return await response.read()
                    else:
                        print(f"Failed to download image: {response.status}")
                        return None
        except Exception as e:
            print(f"Error downloading image: {e}")
            return None
    
    def encode_image_to_base64(self, image_data: bytes) -> str:
        """Encode image data to base64 string"""
        return base64.b64encode(image_data).decode('utf-8')
    
    async def analyze_image(self, image_data: bytes, prompt: str = "Describe this image in detail.") -> str:
        """Analyze an image using a multimodal LLM"""
        # We'll implement multimodal analysis specific to different models
        try:
            # Encode image to base64
            base64_image = self.encode_image_to_base64(image_data)
            
            # Create a message with the image as base64
            message = HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            )
            
            # Get response from the model
            response = await self.llm.ainvoke([message])
            
            return response.content
        except Exception as e:
            print(f"Error analyzing image: {e}")
            return f"I couldn't analyze this image: {str(e)}"
    
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
        return await self.analyze_image(image_data, prompt)
    
    async def extract_text_from_image(self, image_data: bytes) -> str:
        """Extract text from an image (OCR)"""
        # For OCR, we use a specific prompt
        ocr_prompt = "Extract all text visible in this image. Only return the text you see, nothing else."
        
        return await self.analyze_image(image_data, ocr_prompt)

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