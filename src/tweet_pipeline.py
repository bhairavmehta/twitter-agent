import re
import glob
import regex
import ftfy
import logging
import tweepy
import pandas as pd
import os
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
import unicodedata
import time
from rapid_tweepy import RapidTweepy
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()


class TweetPipeline:
    def __init__(self,
                 tweepy_client: tweepy.Client ,
                 rapid_client :RapidTweepy,
                 docs_folder_influencers: str = None,
                 docs_folder_automated: str = None,
                 influencer_csv: str = None,
                 automated_csv: str = None,
                 max_docs: int = 3,
                 lookback : int =24
                 ):
        logging.info("Initializing TweetPipeline...")

        self.docs_folder_influencers = docs_folder_influencers or os.getenv("DOCS_INFLUENCERS")
        self.docs_folder_automated = docs_folder_automated or os.getenv("DOCS_AUTOMATED")

        if not self.docs_folder_influencers or not self.docs_folder_automated:
            logging.error("Both docs folder paths must be provided or set via environment variables.")
            raise ValueError("Both docs folder paths must be provided or set via environment variables.")

        for folder in [self.docs_folder_influencers, self.docs_folder_automated]:
            if not os.path.exists(folder):
                logging.info(f"Creating directory: {folder}")
                os.makedirs(folder)

        self.lookback = lookback
        logging.info(f"Lookback period set to: {self.lookback} hours")

        if not tweepy_client :
            logging.error("A Tweepy client instance must be provided.")
            raise ValueError("A Tweepy client instance must be provided.")


        self.client = tweepy_client
        logging.info("Tweepy client initialized.")
        if not rapid_client :
            logging.error("A rapid client instance must be provided.")
            raise ValueError("A rapid client instance must be provided.")
        self.rapid_client = rapid_client
        self.max_docs = max_docs
        logging.info(f"Maximum documents set to: {self.max_docs}")

        self.user_data = {}
        self.error_usernames = []

        if influencer_csv:
            logging.info(f"Loading influencer data from: {influencer_csv}")
            self.load_users_data_from_csv(influencer_csv, group="influencer")
        if automated_csv:
            logging.info(f"Loading automated data from: {automated_csv}")
            self.load_users_data_from_csv(automated_csv, group="automated")

        logging.info("TweetPipeline initialization complete.")


    @staticmethod
    def extract_username_from_url(url):
        pattern = r'https?://(?:www\.)?[xX]\.com/([A-Za-z0-9_]+)'
        match = re.match(pattern, url)
        return match.group(1) if match else None

    def cache_user_from_url(self, url, group):
        url = str(url).strip()
        username = self.extract_username_from_url(url)
        if username:
            try:
                user_response = self.rapid_client.get_user_info(username=username)
                if user_response:
                    self.user_data[username] = {"user_response": user_response, "group": group}
                else:
                    print(f"User {username} not found.")
                    self.error_usernames.append(username)
            except Exception as e:
                print(f"Error fetching user data for {username}: {str(e)}")
                self.error_usernames.append(username)

    def load_users_data_from_csv(self, csv_file_path, group):
        df = pd.read_csv(csv_file_path,dtype={"id": str})
        if "id" not in df.columns:
            df["id"] = ""

        updated = False
        for idx, row in df.iterrows():
            url = row["Twitter URL"] if "Twitter URL" in df.columns else row['Url']
            username = self.extract_username_from_url(url)
            if not username:
                print(f"Could not extract username from URL: {url}")
                continue
            if pd.isna(row["id"]) or row["id"] == "":
                try:
                    user_response = self.rapid_client.get_user_info(username=username)
                    if user_response:
                        user_id = user_response.id
                        # Cast user_id to string before saving
                        df.at[idx, "id"] = str(user_id)
                        self.user_data[username] = {"user_response": user_response, "group": group}
                        updated = True
                        print(f"Fetched and saved ID for {username}: {str(user_id)}")
                    else:
                        print(f"User {username} not found.")
                        self.error_usernames.append(username)
                except Exception as e:
                    print(f"Error fetching user data for {username}: {str(e)}")
                    self.error_usernames.append(username)
            else:
                dummy_user = type("DummyUser", (), {})()
                dummy_user.id = str(row["id"])
                self.user_data[username] = {"user_response": dummy_user, "group": group}

        if updated:
            df.to_csv(csv_file_path, index=False)
            print(f"CSV file '{csv_file_path}' updated with new user_ids.")

        print(f"Loaded user data for {len(self.user_data)} users.")


    @staticmethod
    def get_start_time(lookback_hours):
        dt = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
        return dt, dt.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'

    def fetch_recent_tweets(self, username, max_results=100):
        tweets_with_context = []
        user_info = self.user_data.get(username)

        if not user_info:
            self.error_usernames.append(username)
            print(f"No cached user data for {username}.")
            return tweets_with_context

        user_id = user_info["user_response"].id
        try:
            tweets = self.rapid_client.get_user_tweets(user_id=user_id,count=max_results)
            for tweet in tweets:
                tweets_with_context.append({"tweet": tweet})
        except Exception as e:
            print(f"Error fetching tweets for {username}: {str(e)}")
            self.error_usernames.append(username)
        return tweets_with_context

    @staticmethod
    def clean_text(text: str) -> str:
        # First fix any obvious text issues
        fixed_text = ftfy.fix_text(text)
        fixed_text = "".join(ch for ch in fixed_text if unicodedata.category(ch)[0] != "C")

        # Remove emojis using an expanded pattern
        emoji_pattern = regex.compile(
            "["
            "\U0001F600-\U0001F64F"  # Emoticons
            "\U0001F300-\U0001F5FF"  # Symbols & pictographs
            "\U0001F680-\U0001F6FF"  # Transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # Flags
            "\U00002702-\U000027B0"  # Dingbats
            "\U000024C2-\U0001F251"  # Enclosed characters
            "\U0001F900-\U0001F9FF"  # Supplemental symbols and pictographs
            "\U0001F000-\U0001F02F"  # Mahjong tiles
            "\U0001FA70-\U0001FAFF"  # Symbols and pictographs extended-A
            "]+",
            flags=regex.UNICODE
        )
        cleaned_text = emoji_pattern.sub("", fixed_text)

        cleaned_text = unicodedata.normalize('NFKC', cleaned_text)

        cleaned_text = cleaned_text.encode('ascii', 'ignore').decode('ascii')

        return cleaned_text

    def _update_doc_files_for_folder(self, folder_path, new_doc_content):
        doc_pattern = folder_path+"/doc_*.txt"
        print(doc_pattern)
        existing_docs = glob.glob(doc_pattern)
        doc_files = []
        for file in existing_docs:
            base = os.path.basename(file)
            try:
                number = int(base.split('_')[1].split('.')[0])
                doc_files.append((number, file))
            except Exception:
                continue
        doc_files.sort(key=lambda x: x[0], reverse=True)

        # Remove existing files if they exceed max_docs
        for number, file_path in doc_files:
            if number >= self.max_docs:
                os.remove(file_path)
                continue
            new_number = number + 1
            new_file_path = os.path.join(folder_path, f"doc_{new_number}.txt")
            os.rename(file_path, new_file_path)

        new_doc_path = os.path.join(folder_path, "doc_1.txt")
        with open(new_doc_path, "w", encoding="utf-8") as f:
            f.write(new_doc_content)

    def update_docs(self):
        influencer_doc = ""
        automated_doc = ""
        for username, info in self.user_data.items():
            print(f"Fetching tweets for {username}...")
            tweets_info = self.fetch_recent_tweets(username)
            for item in tweets_info:
                try:
                    tweet_text = (
                        f"Username: {username}\n"
                        f"Created at: {item['tweet'].created_at.isoformat()}\n"
                        f"Tweet: {item['tweet'].text}\n\n"
                    )
                    tweet_text = self.clean_text(tweet_text)
                    if info["group"] == "influencer":
                        influencer_doc += tweet_text
                    elif info["group"] == "automated":
                        automated_doc += tweet_text
                except Exception as e:
                    print(f"Error processing tweet for {username}: {str(e)}")
                    continue

        # Clean the entire documents one final time
        influencer_doc = self.clean_text(influencer_doc)
        automated_doc = self.clean_text(automated_doc)

        self._update_doc_files_for_folder(self.docs_folder_influencers, influencer_doc)
        self._update_doc_files_for_folder(self.docs_folder_automated, automated_doc)
        print("Document update completed.")
