from google import genai
from google.genai import types
import json
import re
from dotenv import load_dotenv
import os

load_dotenv()

gemini_api_key = os.getenv('GEMINI_API')

client = genai.Client(
    api_key=gemini_api_key,
)


def generate_caption(prompt: str):
    """
    Generate a meme caption based on the given prompt.
    
    Args:
        prompt (str): The user's prompt for meme generation
        
    Returns:
        dict: JSON object with meme data or error information
    """
    response = client.models.generate_content(
        model='gemini-2.0-flash-001',
        contents=f'{prompt}',
        config=types.GenerateContentConfig(
            system_instruction='''Create a viral meme idea with a top and bottom caption.
            The meme should contain only the top and bottom captions, and optionally a middle caption if relevant.
            If the user provides an inappropriate or bad prompt, return an error message.
            Use a relatable tone and humor that will appeal to tech-savvy Gen Z users.
            Please Provide a detail description of the image for the meme_concept.
            
            Output format should be valid JSON with these keys:
            - "meme_concept": Detail description of the image that would fit the captions
            - "top_caption": The top text for the meme
            - "middle_caption": Optional middle text (use null if not needed)
            - "bottom_caption": The bottom text for the meme
            - "error": Optional error message for bad prompts (use null if no error)

            IMPORTANT: Return ONLY valid JSON format. No additional text, explanations, or markdown formatting.
            
            Example structure:
            {
                "meme_concept": "A person sitting at a computer on the left, a robot sitting at a computer on the right",
                "top_caption": "When AI takes over your morning routine",
                "middle_caption": null,
                "bottom_caption": "At least it doesn't spill coffee everywhere",
                "error": null
            }
            
            For bad prompts, return:
            {
                "meme_concept": null,
                "top_caption": null,
                "middle_caption": null,
                "bottom_caption": null,
                "error": "Sorry, I didn't like your prompt"
            }
            ''',
            max_output_tokens=200,
            temperature=0.7,
            top_p=0.9,
            candidate_count=1,
        ),
    )
    
    try:
        # Try to parse the response directly
        result = json.loads(response.text.strip())
        return result
    except json.JSONDecodeError:
        # If direct parsing fails, try to clean the response
        try:
            # Remove markdown code blocks if present
            cleaned_text = re.sub(r'```json\s*|\s*```', '', response.text.strip())
            result = json.loads(cleaned_text)
            return result
        except json.JSONDecodeError:
            # If all parsing fails, return error with raw response for debugging
            return {
                "meme_concept": None,
                "top_caption": None,
                "middle_caption": None,
                "bottom_caption": None,
                "error": "Failed to parse response",
                "raw_response": response.text
            }


# if __name__ == "__main__":
#     # Test the function
#     result = generate_caption("Create a meme where a cat is sleeping in a weird position and show me at the doctor for my back meanwhile how I sleep")
#     print(json.dumps(result, indent=2))