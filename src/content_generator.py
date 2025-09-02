import os
from typing import Optional, Dict
from agents.reply_context_agent import create_reply_context_agent
from agents.reply_composer_agent import create_reply_composer_agent
from agents.comment_context_agent import create_comment_context_agent
from agents.comment_composer_agent import create_comment_composer_agent
from agents.validation_agent import create_validator_agent
import logging
import json
from personality import Personality
from agents.post_generator_agent import create_post_generator_agent
from retrieval_agent import RetrievalAgent
from prompt_analyzer_agent import PromptAnalyzerAgent
from dotenv import load_dotenv
from agents.filter_agent import create_crypto_filter_agent
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class ContentGenerator:
    def __init__(self,
                 personality: Personality = None,
                 api_key: Optional[str] = None,
                 model_name: str = "gpt-4o-mini",
                 post_model_name: str = "o3-mini-high",
                 validation_model_name: str = "gpt-4o-mini",
                 retriever: RetrievalAgent = None,
                 ):

        self.personality = personality
        self.post_model_name = post_model_name
        self.reply_context_agent = create_reply_context_agent(model=model_name, api_key=api_key)
        self.reply_composer_agent = create_reply_composer_agent(model=post_model_name, api_key=api_key,
                                                            reply_examples_file="agents/docs/reply_examples.txt")
        self.post_generator_agent = create_post_generator_agent(model=post_model_name, api_key=api_key)
        self.comment_context_agent = create_comment_context_agent(model=model_name, api_key=api_key)
        self.comment_composer_agent = create_comment_composer_agent(model=post_model_name, api_key=api_key,
                                                            reply_examples_file="agents/docs/reply_examples.txt")

        self.post_validator_agent = create_validator_agent(
            model=validation_model_name, api_key=api_key, markdown=False, show_tool_calls=False, text_type="post"
        )
        self.comment_validator_agent = create_validator_agent(
            model=validation_model_name, api_key=api_key, markdown=False, show_tool_calls=False, text_type="comment"
        )
        self.reply_validator_agent = create_validator_agent(
            model=validation_model_name, api_key=api_key, markdown=False, show_tool_calls=False, text_type="reply"
        )
        self.filter_agent = create_crypto_filter_agent(model=validation_model_name,api_key=api_key)
        # Add the prompt analyzer agent for determining if requests are crypto-related
        self.prompt_analyzer = PromptAnalyzerAgent(model_name="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
        
        if retriever:
            self.retriever = retriever
        else:
            self.retriever = None


    def generate_post(self, current_events: str = "", context: str = "") -> str:
        """Generate a new post based on personality and current context."""
        prompt = f"Event: {current_events}\nContext: {context}"
        if self.retriever:
            retrieved = self.retriever.query(prompt)
            prompt = f"{prompt}\nRelevant Info: {retrieved}"
        post_text = self.post_generator_agent.run(message=prompt, markdown=True, show_tool_calls=True)
        validation_input = f"Text: {post_text}\nContext: {context}"
        validated_post = self.post_validator_agent.run(message=validation_input, markdown=False,
                                                       show_tool_calls=False).content
        return self._format_post(validated_post)


    def generate_comment(self, post_content: str,
                         self_tweet=True, markdown: bool = True,
                         show_tool_calls: bool = True
                         ) -> str:
        """Generate a comment to a tweet using agent workflow."""

        context_response = self.comment_context_agent.run(message=post_content, markdown=markdown,
                                                        show_tool_calls=show_tool_calls)
        comment_context = context_response.content
        if self.retriever:
            retrieved = self.retriever.query(post_content)
            comment_context = f"{comment_context}\n Relevant Info That can be used: {retrieved}"

        combined_input = f"Original Post: {post_content}\n Some context that can be used: {comment_context}"
        if not self_tweet:
            agent = create_reply_composer_agent(self_tweet=self_tweet)
            content = agent.run(message=combined_input)
            return self._format_response(content.content)
        comment_response = self.comment_composer_agent.run(message=combined_input, markdown=markdown,
                                                         show_tool_calls=show_tool_calls,self_tweet=self_tweet)
        tweet_comment = comment_response.content
        """validation_input = f"Text: {tweet_comment}\nContext: {combined_input}"
        validated_comment = self.comment_validator_agent.run(message=validation_input, markdown=markdown,
                                                             show_tool_calls=show_tool_calls,self_tweet=self_tweet)"""
        return self._format_comment(tweet_comment)


    def generate_response(self, input_text: str, self_tweet=True,
                          markdown: bool = True, show_tool_calls: bool = True) -> str:
        """Generate a response to a specific mention."""

        context = self.reply_context_agent.run(message=input_text, markdown=markdown, show_tool_calls=show_tool_calls)
        if self.retriever:
            retrieved = self.retriever.query(input_text)
            context = f"{context}\nRelevant Info: {retrieved}"

        combined_input = f"Original Post/Post with parent tweets: {input_text}\n Context: {context}"
        if not self_tweet:
            agent = create_reply_composer_agent(self_tweet=self_tweet)
            content = agent.run(message=input_text)
            return self._format_response(content.content)
        reply_response = self.reply_composer_agent.run(message=combined_input, markdown=markdown,
                                                       show_tool_calls=show_tool_calls)
        tweet_reply = reply_response.content
        """validation_input = f"Text: {tweet_reply}\nContext: {combined_input}"
        validated_reply = self.reply_validator_agent.run(message=validation_input, markdown=markdown,
                                                         show_tool_calls=show_tool_calls).content"""
        return self._format_response(tweet_reply)
    
    def analyze_user_request(self, user_input: str) -> Dict:
        """
        Analyze a user's request to determine if it's crypto-related and if it asks for media.
        
        Args:
            user_input: The user's input text
            
        Returns:
            A dictionary with analysis and response strategy
        """
        return self.prompt_analyzer.analyze_prompt(user_input)

    def generate_image_prompt(self, input_text: str) -> str:
        """Generate a detailed prompt for image generation based on the input text.
        
        Args:
            input_text: The input text to generate an image prompt from.
            
        Returns:
            A detailed image prompt.
        """
        try:
            # Extract context using the reply context agent
            context_response = self.reply_context_agent.run(message=input_text, markdown=False, show_tool_calls=False)
            context = context_response.content

            # Add additional instructions for enhancing the prompt
            prompt = f"""
            Based on the following input and context, create a detailed and descriptive prompt for AI image generation.
            Make the prompt visually rich with specific details about style, lighting, composition, and subject matter.
            Focus on creating a prompt that will generate a visually appealing and coherent image related to cryptocurrency, 
            finance, or trading. Include specific visual elements that would make the image stand out.
            
            User Input: {input_text}
            
            Additional Context: {context}
            
            Create a detailed image generation prompt (maximum 500 characters):
            """
            
            # Using the reply composer to generate the final image prompt
            response = self.reply_composer_agent.run(message=prompt, markdown=False, show_tool_calls=False)
            image_prompt = response.content.strip()
            
            # Ensure the prompt isn't too long
            if len(image_prompt) > 500:
                image_prompt = image_prompt[:500]
                
            logging.info(f"Generated image prompt: {image_prompt}")
            return image_prompt
        except Exception as e:
            logging.error(f"Error generating image prompt: {e}")
            # Fall back to a simple prompt
            return f"Detailed digital art of cryptocurrency and financial trading related to: {input_text[:200]}"

    def generate_video_prompt(self, input_text: str) -> str:
        """Generate a detailed prompt for video generation based on the input text.
        
        Args:
            input_text: The input text to generate a video prompt from.
            
        Returns:
            A detailed video prompt.
        """
        try:
            # Extract context using the reply context agent
            context_response = self.reply_context_agent.run(message=input_text, markdown=False, show_tool_calls=False)
            context = context_response.content

            # Add additional instructions for enhancing the prompt
            prompt = f"""
            Based on the following input and context, create a detailed and descriptive prompt for AI video generation.
            The prompt should describe a short cinematic sequence with camera movements, lighting, subjects, and atmosphere.
            Focus on creating a prompt that describes a visually appealing high-quality video related to cryptocurrency, 
            blockchain technology, or financial markets. Include specific details about camera angles, motion, and scene composition.
            
            User Input: {input_text}
            
            Additional Context: {context}
            
            Create a detailed video generation prompt (maximum 700 characters):
            """
            
            # Using the reply composer to generate the final video prompt
            response = self.reply_composer_agent.run(message=prompt, markdown=False, show_tool_calls=False)
            video_prompt = response.content.strip()
            
            # Ensure the prompt isn't too long
            if len(video_prompt) > 700:
                video_prompt = video_prompt[:700]
                
            logging.info(f"Generated video prompt: {video_prompt}")
            return video_prompt
        except Exception as e:
            logging.error(f"Error generating video prompt: {e}")
            return f"A cinematic sequence showing cryptocurrency trading and blockchain technology related to: {input_text[:200]}"

    def filter_comment(self, comment_context: str, markdown: bool = True,
                                     show_tool_calls: bool = False) -> bool:
        """
        Evaluate if a comment is worth replying to based on its cryptocurrency relevance.

        Args:
            comment_context: The context of the comment to be evaluated.
            markdown: Whether the response should use markdown formatting.
            show_tool_calls: Whether to show tool call logs.

        Returns:
            A boolean value indicating if the comment is worth replying to.
            Returns False if an error occurs or if the filter response is not as expected.
        """
        try:
            # Run the filter agent on the provided comment context.
            response = self.filter_agent.run(message=comment_context,markdown=markdown)
            result = json.loads(response.content)
            if isinstance(result, dict) and "should_reply" in result:
                return bool(result["should_reply"])
            else:
                logging.error("Filter agent response did not contain 'should_reply' key.")
                return False
        except Exception as e:
            logging.error(f"Error checking comment for reply: {e}")
            return False

    @staticmethod
    def _format_text(content: str) -> str:
        """Safeguard to ensure Twitter's character limit is not exceeded."""
        content = content.strip("'\"")
        content = content.strip()
        if len(content) > 280:
            content = content[:277] + "..."
        return content

    def _format_response(self, content: str) -> str:
        """Format response text."""
        return self._format_text(content)

    def _format_post(self, content: str) -> str:
        """Format post text."""
        return self._format_text(content)

    def _format_comment(self, content: str) -> str:
        """Format comment text."""
        return self._format_text(content)