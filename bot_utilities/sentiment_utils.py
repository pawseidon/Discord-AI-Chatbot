import aiohttp
import json
from typing import Dict, Any, List, Tuple
from bot_utilities.config_loader import config

class SentimentAnalyzer:
    def __init__(self):
        """Initialize the sentiment analyzer."""
        # Model configuration
        self.api_base_url = config.get('API_BASE_URL', 'https://api.groq.com/openai/v1')
        self.api_key = config.get('API_KEY', '')
        self.model_id = config.get('MODEL_ID', 'llama-3.1-8b-instant')
    
    async def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze the sentiment of the provided text.
        
        Args:
            text (str): The text to analyze
            
        Returns:
            Dict[str, Any]: A dictionary containing sentiment analysis results including:
                - sentiment: overall sentiment (positive, negative, neutral)
                - confidence: confidence level of the analysis
                - emotions: detected emotions and their intensities
                - summary: brief summary of the sentiment analysis
        """
        try:
            # Prepare the system prompt for sentiment analysis
            system_prompt = """You are a sentiment analysis expert. Analyze the following text and provide a JSON response with these fields:
            1. sentiment: The overall sentiment as "positive", "negative", or "neutral"
            2. confidence: A number from 0.0 to 1.0 indicating confidence in the assessment
            3. emotions: An object with emotions detected (joy, sadness, anger, fear, surprise, disgust) and their intensity from 0.0 to 1.0
            4. summary: A brief one-sentence summary of the sentiment analysis

            Format your response as valid JSON only, with no additional text.
            """
            
            # Make API request to the language model
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            data = {
                "model": self.model_id,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text}
                ],
                "response_format": {"type": "json_object"},
                "temperature": 0.2,  # Lower temperature for more consistent results
                "max_tokens": 500
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.api_base_url}/chat/completions",
                    headers=headers,
                    json=data
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        return {
                            "error": f"API request failed with status {response.status}",
                            "details": error_text
                        }
                    
                    result = await response.json()
                    
                    # Extract the content from the API response
                    if "choices" in result and len(result["choices"]) > 0:
                        content = result["choices"][0]["message"]["content"]
                        
                        # Parse the JSON content
                        try:
                            sentiment_data = json.loads(content)
                            return sentiment_data
                        except json.JSONDecodeError:
                            # If JSON parsing fails, extract what we can
                            return {
                                "sentiment": "unknown",
                                "confidence": 0.0,
                                "emotions": {},
                                "summary": "Failed to parse sentiment analysis result",
                                "raw_content": content
                            }
            
        except Exception as e:
            return {
                "sentiment": "unknown",
                "confidence": 0.0,
                "emotions": {},
                "summary": f"Error during sentiment analysis: {str(e)}"
            }
    
    async def get_sentiment_emoji(self, sentiment: str) -> str:
        """
        Get an appropriate emoji for the given sentiment.
        
        Args:
            sentiment (str): The sentiment (positive, negative, neutral)
            
        Returns:
            str: An emoji representing the sentiment
        """
        sentiment_emojis = {
            "positive": "😊",
            "very positive": "😄",
            "negative": "😞",
            "very negative": "😠",
            "neutral": "😐",
            "mixed": "😕",
            "unknown": "❓"
        }
        
        return sentiment_emojis.get(sentiment.lower(), "❓")
    
    async def get_emotion_emojis(self, emotions: Dict[str, float]) -> List[str]:
        """
        Get appropriate emojis for the detected emotions.
        
        Args:
            emotions (Dict[str, float]): Dictionary of emotions and their intensities
            
        Returns:
            List[str]: List of emotion emojis
        """
        emotion_emojis = {
            "joy": "😄",
            "happiness": "😊",
            "sadness": "😢",
            "anger": "😠",
            "fear": "😨",
            "surprise": "😮",
            "disgust": "🤢",
            "love": "❤️",
            "excitement": "🤩",
            "confusion": "😕",
            "disappointment": "😔",
            "anxiety": "😰",
            "gratitude": "🙏",
            "pride": "😌"
        }
        
        # Filter for emotions with significant intensity (> 0.3)
        significant_emotions = [emotion for emotion, intensity in emotions.items() 
                               if intensity > 0.3]
        
        # Get emojis for significant emotions
        emotion_emoji_list = [emotion_emojis.get(emotion.lower(), "") 
                             for emotion in significant_emotions
                             if emotion.lower() in emotion_emojis]
        
        return emotion_emoji_list
    
    async def format_sentiment_analysis(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format sentiment analysis results for display.
        
        Args:
            analysis (Dict[str, Any]): The sentiment analysis result
            
        Returns:
            Dict[str, Any]: Formatted sentiment analysis with display-friendly fields
        """
        sentiment = analysis.get("sentiment", "unknown")
        emoji = await self.get_sentiment_emoji(sentiment)
        
        emotions = analysis.get("emotions", {})
        emotion_emojis = await self.get_emotion_emojis(emotions)
        
        # Calculate confidence percentage
        confidence = analysis.get("confidence", 0) * 100
        
        # Format emotions for display
        formatted_emotions = []
        for emotion, intensity in emotions.items():
            if intensity > 0.2:  # Only show significant emotions
                emoji = await self.get_sentiment_emoji(emotion)
                formatted_emotions.append(f"{emoji} {emotion.capitalize()}: {int(intensity * 100)}%")
        
        return {
            "sentiment": sentiment.capitalize(),
            "sentiment_emoji": emoji,
            "confidence": f"{confidence:.1f}%",
            "emotion_emojis": " ".join(emotion_emojis),
            "formatted_emotions": formatted_emotions,
            "summary": analysis.get("summary", "No summary available")
        } 