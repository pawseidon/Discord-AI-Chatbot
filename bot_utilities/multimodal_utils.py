import os
import io
import base64
import aiohttp
import asyncio
from typing import List, Optional, Dict, Any, Union
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