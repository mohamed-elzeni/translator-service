import openai
import time
import os

openai.api_key = os.environ.get("OPENAI_API_KEY")
openai.api_base = "https://elzeni-recitation-resource.openai.azure.com/"
openai.api_type = "azure"
openai.api_version = "2024-08-01-preview"
deployment_name = "elzeni-gpt-4"


def translate_content(content: str) -> tuple[bool, str]:
    context = """
    You are a helpful assistant. Your task is to:
    1. Identify the language of the following text.
    2. Translate the text to English if it is not in English.
    3. If the text is already in English, output the text as it is.
    4. If the text is unintelligible or malformed, output only an empty string.
    Provide your response in the format: "Language: <language>, Translation: <translation>"
    """

    retry_delay = 9

    while True:
        try:
            # Call the OpenAI API
            response = openai.ChatCompletion.create(
                engine=deployment_name,
                messages=[
                    {"role": "system", "content": context},
                    {"role": "user", "content": content},
                ],
            )

            # Extract and process the response
            response_text = response.choices[0].message.content.strip()
            language, translation = response_text.split(", Translation: ")

            # Determine if the original post was in English
            is_english = language.lower() == "language: english"

            return (is_english, translation)

        except openai.error.RateLimitError:
            time.sleep(retry_delay)

        except Exception as e:
            print(f"An error occurred: {e}")
            return (False, "")
