import os
import google.generativeai as genai
from typing import List, Tuple, Dict, Any

# Attempt to configure Gemini API key from environment variable
# Ensure GEMINI_API_KEY is set in your environment
try:
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set.")
    genai.configure(api_key=gemini_api_key)
except ValueError as e:
    print(f"Error configuring Gemini API: {e}")
    # You might want to handle this more gracefully or exit
    # For now, we'll let it proceed, but API calls will fail if not configured

# Placeholder for retry decorator if needed, similar to the tenacity retry in the original code
# For now, we'll implement a simple call.
def call_gemini_api(
    prompt_text: str,
    stop_sequences: List[str] = None, # Made optional, as Gemini might not always need it or handles it differently
    candidate_count: int = 1,
    temperature: float = 0.7, # Default temperature
    max_output_tokens: int = 256, # Default max tokens
    top_p: float = None, # Optional
    model_name: str = 'gemini-pro' # Default model
) -> Tuple[Tuple[str, float], List[Tuple[str, float]]]:
    """
    Calls the Gemini API and processes the response.

    Args:
        prompt_text: The prompt to send to the Gemini API.
        stop_sequences: A list of sequences to stop generation at.
        candidate_count: The number of candidate responses to generate.
        temperature: The sampling temperature.
        max_output_tokens: The maximum number of tokens to generate.
        top_p: The nucleus sampling probability.
        model_name: The name of the Gemini model to use.

    Returns:
        A tuple containing:
        - The best response (text, score/probability).
        - A list of all responses [(text, score/probability), ...].
        Returns (('skip', 0.0), []) if an error occurs or no valid response.
    """
    if not genai.API_KEY: # Check if API key was successfully configured
        print("Gemini API key not configured. Cannot make API calls.")
        return ('skip', 0.0), []

    try:
        model = genai.GenerativeModel(model_name)

        generation_config_params = {
            "candidate_count": candidate_count,
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
        }
        if stop_sequences:
            generation_config_params["stop_sequences"] = stop_sequences
        if top_p is not None:
            generation_config_params["top_p"] = top_p

        generation_config = genai.types.GenerationConfig(**generation_config_params)
        
        response = model.generate_content(prompt_text, generation_config=generation_config)

        response_list: List[Tuple[str, float]] = []
        if response.candidates:
            for candidate in response.candidates:
                # Gemini's candidate object has 'text'.
                # It doesn't directly provide logprobs like OpenAI Davinci.
                # We'll use a placeholder score (e.g., 1.0 for the first, or based on order).
                # For now, let's assign a decreasing score based on order if multiple candidates.
                # This is a simplification; actual scoring might need more thought.
                candidate_text = ''.join(part.text for part in candidate.content.parts)
                
                # Placeholder for score - higher for earlier candidates if multiple
                score = 1.0 / (len(response_list) + 1) if candidate_count > 1 else 1.0
                response_list.append((candidate_text.strip(), score))
        
        if not response_list:
            print("Gemini API returned no candidates.")
            return ('skip', 0.0), []

        # The original code sorts by probability if n > 1.
        # Our placeholder score already implies order if Gemini returns them ordered.
        # If not, explicit sorting might be needed if a better score heuristic is developed.
        # For now, we assume the first candidate is the "best" if multiple are returned.

        return response_list[0], response_list

    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return ('skip', 0.0), []

if __name__ == '__main__':
    # Example usage (requires GEMINI_API_KEY to be set)
    if genai.API_KEY:
        print("Testing Gemini API call...")
        prompt = "Translate 'hello world' into French."
        best_response, all_responses = call_gemini_api(prompt, candidate_count=1, temperature=0.5)
        
        if best_response[0] != 'skip':
            print(f"Best response: {best_response[0]} (Score: {best_response[1]})")
            if len(all_responses) > 1:
                print("All responses:")
                for i, (text, score) in enumerate(all_responses):
                    print(f"  {i+1}. {text} (Score: {score})")
        else:
            print("Gemini API call failed or returned no valid response.")
    else:
        print("Skipping Gemini API test as API key is not configured.")
