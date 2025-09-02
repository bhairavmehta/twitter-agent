import asyncio
import logging
import requests
import tempfile
import os
import time
from typing import Optional, Dict, Any, Union, Literal
import tweepy
import fal_client
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class MediaGenerator:
    """
    Class to handle image and video generation using the fal.ai API and posting to Twitter.
    """
    def __init__(
        self,
        image_model: str = "fal-ai/flux-pro/v1.1-ultra",
        video_model: str = "fal-ai/veo2"
    ):
        """
        Initialize the media generator.
        
        Args:
            image_model: The model to use for image generation.
            video_model: The model to use for video generation.
        """
        self.image_model = image_model
        self.video_model = video_model
        logging.info(f"Initialized MediaGenerator with image model {image_model} and video model {video_model}")

        fal_api_key = os.getenv("FAL_API_KEY")
        if not fal_api_key:
            raise ValueError("FAL_API_KEY not found in environment variables")

        os.environ['FAL_KEY'] = fal_api_key

        # Set up Twitter v1 API for media uploads
        auth = tweepy.OAuth1UserHandler(
            consumer_key=os.getenv('TWITTER_API_KEY'),
            consumer_secret=os.getenv('TWITTER_API_SECRET'),
            access_token=os.getenv('TWITTER_ACCESS_TOKEN'),
            access_token_secret=os.getenv('TWITTER_ACCESS_TOKEN_SECRET'),
        )
        self.twitter_api_v1 = tweepy.API(auth)
    
    async def generate_image(self, prompt: str) -> Optional[Dict[str, Any]]:
        """
        Generate an image based on the given prompt.
        
        Args:
            prompt: The prompt to generate an image from.
            
        Returns:
            A dictionary containing the image URL and other metadata, or None if generation failed.
        """
        try:
            logging.info(f"Generating image with prompt: {prompt}")
            handler = await fal_client.submit_async(
                self.image_model,
                arguments={
                    "prompt": prompt
                },
            )
            
            # Log generation events
            async for event in handler.iter_events(with_logs=True):
                logging.debug(f"Image generation event: {event}")
            
            # Wait for completion
            result = await handler.get()
            logging.info(f"Generated image successfully")
            return result
        except Exception as e:
            logging.error(f"Error generating image: {e}")
            return None

    async def generate_video(self, prompt: str) -> Optional[Dict[str, Any]]:
        """
        Generate a video based on the given prompt.
        
        Args:
            prompt: The prompt to generate a video from.
            
        Returns:
            A dictionary containing the video URL and other metadata, or None if generation failed.
        """
        try:
            logging.info(f"Generating video with prompt: {prompt}")
            
            # Define a callback to handle progress updates
            def on_queue_update(update):
                if isinstance(update, fal_client.InProgress):
                    for log in update.logs:
                        logging.info(f"Video generation progress: {log['message']}")
            
            # Submit the generation request
            result = await fal_client.subscribe_async(
                self.video_model,
                arguments={
                    "prompt": prompt
                },
                with_logs=True,
                on_queue_update=on_queue_update,
            )
            print(result)
            logging.info(f"Generated video successfully")
            return result
        except Exception as e:
            logging.error(f"Error generating video: {e}")
            return None
            
    def download_media(self, media_url: str, media_type: Literal["image", "video"]) -> Optional[str]:
        """
        Download media from the given URL and save it to a temporary file.
        
        Args:
            media_url: The URL of the media to download.
            media_type: The type of media, either "image" or "video".
            
        Returns:
            The path to the downloaded media, or None if download failed.
        """
        try:
            response = requests.get(media_url, stream=True)
            if response.status_code != 200:
                logging.error(f"Failed to download media, status code: {response.status_code}")
                return None
                
            # Create a temporary file with the appropriate extension
            extension = ".mp4" if media_type == "video" else ".jpg"
            fd, temp_path = tempfile.mkstemp(suffix=extension)
            with os.fdopen(fd, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    
            logging.info(f"Downloaded {media_type} to {temp_path}")
            return temp_path
        except Exception as e:
            logging.error(f"Error downloading {media_type}: {e}")
            return None
            
    async def generate_and_upload_image(self, prompt: str) -> Optional[str]:
        """
        Generate an image based on the prompt, download it, and upload it to Twitter.
        
        Args:
            prompt: The prompt to generate an image from.
            
        Returns:
            The media ID string if successful, None otherwise.
        """
        try:
            # Generate the image
            image_result = await self.generate_image(prompt)
            print(image_result)
            if not image_result or "images" not in image_result or not image_result["images"]:
                logging.error("Failed to generate image")
                return None
            
            # Download the image
            image_url = image_result["images"][0]["url"]
            image_path = self.download_media(image_url, "image")
            
            if not image_path:
                logging.error("Failed to download image")
                return None
                
            try:
                # Upload the image to Twitter
                media = self.twitter_api_v1.media_upload(filename=image_path)
                media_id = media.media_id_string
                logging.info(f"Uploaded image to Twitter with media ID: {media_id}")
                
                # Clean up the temporary file
                os.remove(image_path)
                
                return media_id
            except Exception as e:
                logging.error(f"Error uploading image to Twitter: {e}")
                # Clean up the temporary file
                if image_path and os.path.exists(image_path):
                    os.remove(image_path)
                return None
                
        except Exception as e:
            logging.error(f"Error in generate_and_upload_image: {e}")
            return None
    
    async def generate_and_upload_video(self, prompt: str) -> Optional[str]:
        """
        Generate a video based on the prompt, download it, and upload it to Twitter.
        
        Args:
            prompt: The prompt to generate a video from.
            
        Returns:
            The media ID string if successful, None otherwise.
        """
        try:
            # Generate the video
            video_result = await self.generate_video(prompt)

            if not video_result or "video" not in video_result:
                logging.error("Failed to generate video")
                return None
            
            # Download the video
            video_url = video_result["video"]["url"]
            video_path = self.download_media(video_url, "video")

            if not video_path:
                logging.error("Failed to download video")
                return None
                
            try:
                media = self.twitter_api_v1.media_upload(
                    filename=video_path,
                    media_category='tweet_video'
                )

                if hasattr(media, 'processing_info'):
                    self._wait_for_video_processing(media)
                
                media_id = media.media_id_string
                logging.info(f"Uploaded video to Twitter with media ID: {media_id}")
                
                # Clean up the temporary file
                os.remove(video_path)
                
                return media_id
            except Exception as e:
                logging.error(f"Error uploading video to Twitter: {e}")
                # Clean up the temporary file
                if video_path and os.path.exists(video_path):
                    os.remove(video_path)
                return None
                
        except Exception as e:
            logging.error(f"Error in generate_and_upload_video: {e}")
            return None
            
    def _wait_for_video_processing(self, media_upload_result):
        """
        Wait for a video to finish processing on Twitter.
        
        Args:
            media_upload_result: The result from the media_upload call.
        """
        processing_info = media_upload_result.processing_info
        
        if processing_info is None or processing_info['state'] == 'succeeded':
            return
            
        if processing_info['state'] == 'failed':
            logging.error(f"Video processing failed: {processing_info['error']}")
            return
            
        # If we're still processing, wait and check again
        check_after_secs = processing_info.get('check_after_secs', 5)
        logging.info(f"Waiting {check_after_secs} seconds for video processing...")
        time.sleep(check_after_secs)
        
        # Get the updated status
        media_id = media_upload_result.media_id
        status = self.twitter_api_v1.get_media_upload_status(media_id=media_id)
        
        # Recursively check again
        self._wait_for_video_processing(status)
