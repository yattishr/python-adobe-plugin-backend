import pandas as pd
import google.generativeai as genai
import time
import random
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setting up the API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is not set in the environment variables")


def read_csv_data(file_path, sample_size=1000):
    print("Reading and sampling csv file....")
    df = pd.read_csv(file_path)
    return df.sample(min(sample_size, len(df)))


def dataframe_to_text(df):
    print("Converting pandas dataframe to text....")
    return df.to_csv(index=False)


def send_data_to_gemini(data, description, model_name="gemini-1.5-pro", max_retries=3):
    print("Sending data to Gemini....")

    genai.configure(api_key=GOOGLE_API_KEY)

    system_prompt = """You are an expert data analyst who fully understands how to present data in story form
    that is easier for users to understand their data. Your role is to create a compelling story based on
     the data that is provided to highlight key aspects and components from the data.
     In addition, you will highlight any identifiable trends you recognize from the data and present it
     in a relatable story that the user can relate to.\n\n
     Your task is to analyze the data and provide an aggregated view of the data in a compelling story form."""

    generation_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8192,
    }

    model = genai.GenerativeModel(
        model_name=model_name, generation_config=generation_config
    )

    prompt = f"Analyze the following CSV data:\n```\nDescription of the data: {description}\n```\n{data}\n```\n{system_prompt}"

    time.sleep(2)
    for attempt in range(max_retries):
        try:
            print(f"Sending the prompt to the model (attempt {attempt + 1})...")
            chat = model.start_chat(history=[])

            # Try with streaming first
            try:
                response = chat.send_message(prompt, stream=True)
                full_response = ""
                for chunk in response:
                    full_response += chunk.text
                    print(chunk.text, end="", flush=True)
            except Exception as stream_error:
                print(
                    f"Streaming failed: {stream_error}. Retrying without streaming..."
                )
                # If streaming fails, try without streaming
                response = chat.send_message(prompt, stream=False)
                full_response = response.text
                print(full_response)

            print("\nResponse completed.")
            return full_response

        except genai.types.generation_types.BlockedPromptException as e:
            print(f"Blocked prompt: {e}")
            return None
        except Exception as e:
            print(f"An error occurred: {e}")
            if attempt < max_retries - 1:
                wait_time = (2**attempt) + random.uniform(0, 1)
                print(f"Retrying in {wait_time:.2f} seconds...")
                time.sleep(wait_time)
            else:
                print("Max retries reached. Exiting.")
                return None


def main():
    print("Running from main....")
    # file_path = ".\data\Churn_Modelling_Small.csv"
    # description = """This dataset contains information about Fictituous Bank customers and whether they have churned (left the bank) or not.
    # The columns include customer ID, surname, credit score, geography, gender, age, tenure, balance, number of products,
    # whether they have a credit card, whether they are an active member, estimated salary, and whether they churned."""

    file_path = ".\data\Mall_Customers.csv"
    description = """This dataset contains information about customers of a mall.
    The columns include CustomerID, Genre (Gender), Age, Annual Income (k$), Spending Score (1-100)."""

    # file_path = ".//data//50_Startups.csv"
    # description = """This dataset contains financial information about 50 startup companies.
    # The columns include R&D Spend, Administration, Marketing Spend, State, Profit"""

    df = read_csv_data(file_path, sample_size=1000)  # Adjust sample size as needed
    data_text = dataframe_to_text(df)
    response = send_data_to_gemini(data_text, description)
    if response:
        print("\nFinal Response:")
        print(response)


if __name__ == "__main__":
    main()
