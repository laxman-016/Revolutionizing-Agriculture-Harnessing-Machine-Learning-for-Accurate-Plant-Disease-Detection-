
import openai

# Set your OpenAI API key
openai.api_key = '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'

# Define a prompt or message for the chatbot
prompt = "imformation related to plants ,crop related dieases and nutrients and their prescribtion with fertilizers and pesticides"

# Make a request to the OpenAI API
response = openai.Completion.create(
  engine="text-davinci-003",  # Use the appropriate engine
  prompt=prompt,
  max_tokens=100  # Adjust the max tokens based on your needs
)

# Extract the generated text from the OpenAI response
generated_text = response['choices'][0]['text']

# Print or use the generated text for training your chatbot
print(generated_text)
