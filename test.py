from google.genai.types import Tool, GenerateContentConfig, GoogleSearch
from google import genai

api_key = "GOOGLE-API-KEY"
client = genai.Client(api_key=api_key)

model_id = "gemini-2.0-flash"

google_search_tool = Tool(
    google_search = GoogleSearch()
)

image = client.files.upload(file="can.jpeg")

response = client.models.generate_content(
    model=model_id,
    contents=[image, "How many calories are in this?"],
    config=GenerateContentConfig(
        tools=[google_search_tool],
        response_modalities=["TEXT"],
    )
)

for each in response.candidates[0].content.parts:
    print(each.text)

