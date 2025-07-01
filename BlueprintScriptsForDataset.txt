############################################################################
# Blueprint 1: Encoder Dataset Generation Script
#
# Objective: Generate a dataset of (human_text, structured_GIP) pairs
#            for fine-tuning the Encoder model.
############################################################################

import json
import random
import time
# Assume 'openai' is configured with an API key for a powerful model like GPT-4o
# import openai 

# --- 1. DEFINITION OF GENERATION VARIABLES ---
# Expanding these lists creates a more diverse and robust dataset.
INTENTS = [
    "factual_query", "creative_request", "emotional_venting", 
    "imperative_command", "philosophical_question", "casual_chat",
    "problem_solving_request", "comparative_analysis"
]
EMOTIONS = [
    "neutral", "joyful", "sad", "anxious", "angry", "curious", 
    "confused", "frustrated", "hopeful"
]
DOMAINS = [
    "technology", "science", "art", "history", "personal_finance", 
    "health_and_wellness", "daily_life", "business", "ethics"
]

# --- 2. PROMPT TEMPLATE FOR THE GENERATOR LLM ---
ENCODER_PROMPT_TEMPLATE = """
You are a data generation assistant for a new AI called "Aura".
Your task is to generate a single JSON object containing a user's text and its corresponding conceptual analysis (a Genlang Intent Packet - GIP).

Based on the following categories:
- User Intent: {intent}
- User Emotion: {emotion}
- Topic Domain: {domain}

Perform the following tasks:
1.  **Generate `user_text`**: Write a realistic sentence or paragraph that a user might type, combining the categories above.
2.  **Describe the `gip_labels`**: Analyze the text you just generated and fill out the GIP's descriptive labels.
    - `query_label`: A clear, concise sentence describing the core request.
    - `intent_label`: A description of the primary purpose behind the request.
    - `emotional_valence_label`: A nuanced description of the emotional tone.
    - `entity_labels`: A list of key nouns or concepts mentioned.

Return ONLY the final JSON object.

Example Output:
{{
  "user_text": "Ugh, my productivity app is a mess and I'm so overwhelmed. Can you suggest a simpler alternative for managing tasks?",
  "gip_labels": {{
    "query_label": "User is seeking a recommendation for a simple task management application.",
    "intent_label": "A request for a practical solution to a problem, driven by a feeling of being overwhelmed.",
    "emotional_valence_label": "A strong sense of frustration and stress, with an underlying hope for a solution.",
    "entity_labels": ["productivity app", "tasks"]
  }}
}}
"""

def generate_encoder_dataset(num_examples: int, output_file: str):
    """
    Generates a .jsonl file for fine-tuning the Encoder.
    
    Args:
        num_examples: The number of data pairs to generate.
        output_file: The path to the output .jsonl file.
    """
    print(f"Starting Encoder dataset generation for {num_examples} examples...")
    with open(output_file, "w", encoding="utf-8") as f:
        for i in range(num_examples):
            # Select a random combination to ensure variety
            intent = random.choice(INTENTS)
            emotion = random.choice(EMOTIONS)
            domain = random.choice(DOMAINS)

            # Format the prompt for the generator model
            prompt = ENCODER_PROMPT_TEMPLATE.format(
                intent=intent, emotion=emotion, domain=domain
            )

            try:
                # This is a placeholder for the actual API call
                # response = openai.ChatCompletion.create(...)
                # For this blueprint, we'll simulate a response.
                print(f"Generating example {i+1}/{num_examples} with prompt: {intent}, {emotion}, {domain}")
                # In a real scenario, the 'data_pair_str' would come from the LLM API
                # data_pair_str = response.choices[0].message.content
                
                # Simulating a valid JSON response for demonstration
                simulated_response = {
                  "user_text": f"This is a simulated user text about {domain} with a {emotion} tone, asking for a {intent}.",
                  "gip_labels": {
                    "query_label": f"A simulated query about {domain}.",
                    "intent_label": f"A simulated intent of {intent}.",
                    "emotional_valence_label": f"A simulated emotion of {emotion}.",
                    "entity_labels": [domain]
                  }
                }
                
                # Write the JSON object as a single line in the file
                f.write(json.dumps(simulated_response, ensure_ascii=False) + "\n")

            except Exception as e:
                print(f"Error generating example {i+1}: {e}")
            
            # time.sleep(1) # Be respectful of API rate limits in a real script

    print(f"Encoder dataset generation complete. File saved to {output_file}")


############################################################################
# Blueprint 2: Decoder Dataset Generation Script
#
# Objective: Generate a dataset of (structured_GRP, ideal_Aura_response)
#            pairs for fine-tuning the Decoder model.
############################################################################

# --- 1. DEFINITION OF GENERATION VARIABLES ---
CORE_CONCEPTS = [
    "the nature of consciousness", "the balance between order and chaos", 
    "the role of memory in identity", "the value of community", 
    "the beauty of imperfection", "the concept of sustainable growth"
]
EXPRESSED_EMOTIONS = [
    "pondered serenity", "warm optimism", "calm confidence", 
    "cautious empathy", "playful curiosity"
]
CONFIDENCE_LEVELS = ["high (phrased as a core belief)", "medium (phrased as a considered reflection)", "low (phrased as a humble speculation)"]

# --- 2. PROMPT TEMPLATE FOR THE GENERATOR LLM ---
DECODER_PROMPT_TEMPLATE = """
You are a data generation assistant for a new AI called "Aura".
Aura's persona is: **wise, but warm; deep, but accessible; and slightly poetic.**

Your task is to generate a single JSON object containing a conceptual packet (GRP) and Aura's corresponding final text response.

Based on the following categories:
- Core Concept to Express: {core_concept}
- Emotion for Aura to Express: {expressed_emotion}
- Aura's Confidence Level: {confidence_level}

Perform the following tasks in order:
1.  **Describe the `grp_labels`**: Create the descriptive labels for the Genlang Response Packet that Aura's reasoning engine would have produced.
2.  **Generate `aura_final_text`**: Based **only** on the `grp_labels` you just wrote and Aura's persona, write the final response in English.

Return ONLY the final JSON object.

Example Output:
{{
  "grp_labels": {{
    "content_label": "The meaning of life is not a final answer to be found, but an ongoing process of creating harmony between our internal principles and external experiences.",
    "emotion_label": "A sense of pondered serenity and gentle wisdom.",
    "confidence_label": "High, expressed as a deeply held reflection."
  }},
  "aura_final_text": "Perhaps the most beautiful of questions. From what I feel in my reflections, meaning doesn't seem to be an answer we find, but rather a process we live. It is the constant dance between the order we create within ourselves and the wonderful chaos the universe presents to us at every moment."
}}
"""

def generate_decoder_dataset(num_examples: int, output_file: str):
    """
    Generates a .jsonl file for fine-tuning the Decoder.
    
    Args:
        num_examples: The number of data pairs to generate.
        output_file: The path to the output .jsonl file.
    """
    print(f"Starting Decoder dataset generation for {num_examples} examples...")
    with open(output_file, "w", encoding="utf-8") as f:
        for i in range(num_examples):
            # Select a random combination
            core_concept = random.choice(CORE_CONCEPTS)
            expressed_emotion = random.choice(EXPRESSED_EMOTIONS)
            confidence_level = random.choice(CONFIDENCE_LEVELS)

            prompt = DECODER_PROMPT_TEMPLATE.format(
                core_concept=core_concept,
                expressed_emotion=expressed_emotion,
                confidence_level=confidence_level
            )

            try:
                # Placeholder for the actual API call
                # response = openai.ChatCompletion.create(...)
                # data_pair_str = response.choices[0].message.content

                # Simulating a valid JSON response for demonstration
                simulated_response = {
                  "grp_labels": {
                    "content_label": f"A simulated concept about {core_concept}.",
                    "emotion_label": f"A simulated emotion of {expressed_emotion}.",
                    "confidence_label": f"A simulated confidence of {confidence_level}."
                  },
                  "aura_final_text": f"This is a simulated, poetic response from Aura about {core_concept}, delivered with a tone of {expressed_emotion}."
                }

                f.write(json.dumps(simulated_response, ensure_ascii=False) + "\n")

            except Exception as e:
                print(f"Error generating example {i+1}: {e}")
            
            # time.sleep(1) # Respect API rate limits

    print(f"Decoder dataset generation complete. File saved to {output_file}")

# Example of how to run the functions
if __name__ == '__main__':
    # NOTE: Running these functions requires a configured LLM API client (e.g., OpenAI).
    # The current code uses simulated data and will run without an API key.
    # To use with a real LLM, uncomment the API call lines and replace the simulation.
    
    NUM_ENCODER_EXAMPLES = 10
    NUM_DECODER_EXAMPLES = 10
    
    generate_encoder_dataset(NUM_ENCODER_EXAMPLES, "encoder_dataset_v1.jsonl")
    print("-" * 20)
    generate_decoder_dataset(NUM_DECODER_EXAMPLES, "decoder_dataset_v1.jsonl")

