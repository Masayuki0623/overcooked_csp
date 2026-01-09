import os
import json
from openai import OpenAI
import google.generativeai as genai

class LLMService:
    def __init__(self, api_key=None, model="gpt-4o-mini"):
        self.model = model
        self.client = None
        self.is_gemini = self.model.startswith("gemini")

        if self.is_gemini:
            self.api_key = api_key or os.getenv("GOOGLE_API_KEY") or self._read_key_from_file("google_api_key.txt")
            if self.api_key:
                genai.configure(api_key=self.api_key)
            else:
                print("[LLMService] Warning: No API key provided and GOOGLE_API_KEY not set for Gemini.")
        else:
            self.api_key = api_key or os.getenv("OPENAI_API_KEY") or self._read_key_from_file("openai_api_key.txt")
            if self.api_key:
                self.client = OpenAI(api_key=self.api_key)
            else:
                print("[LLMService] Warning: No API key provided and OPENAI_API_KEY not set for OpenAI.")

    def _read_key_from_file(self, filename):
        try:
            if os.path.exists(filename):
                with open(filename, 'r', encoding='utf-8') as f:
                    return f.read().strip()
        except Exception:
            pass
        return None

    def set_model(self, model_name):
        self.model = model_name
        self.is_gemini = self.model.startswith("gemini")
        # Re-initialize client/config if switching provider?
        # For simplicity, assuming __init__ is called with correct model or env vars are set for both if switching dynamically.
        if self.is_gemini:
            if not os.getenv("GOOGLE_API_KEY"):
                 print("[LLMService] Warning: Switched to Gemini but GOOGLE_API_KEY is not set.")
        else:
            if not self.client and os.getenv("OPENAI_API_KEY"):
                self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def infer_weights(self, tasks, instruction, system_prompt_path):
        if self.is_gemini:
             if not os.getenv("GOOGLE_API_KEY") and not self.api_key:
                 return {"error": "Google API Key missing"}
        elif not self.client:
            return {"error": "OpenAI API Key missing"}

        try:
            with open(system_prompt_path, 'r', encoding='utf-8') as f:
                system_prompt = f.read()
        except Exception as e:
            return {"error": f"Failed to read system prompt: {e}"}

        user_content = f"Tasks: {json.dumps(tasks)}\nInstruction: \"{instruction}\""

        try:
            if self.is_gemini:
                # Gemini Implementation
                model_name = self.model
                if not model_name.startswith("models/"):
                    model_name = f"models/{model_name}"
                
                model = genai.GenerativeModel(
                    model_name,
                    system_instruction=system_prompt,
                    generation_config={"response_mime_type": "application/json"}
                )
                response = model.generate_content(user_content)
                content = response.text
            else:
                # OpenAI Implementation
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ]
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.1,
                    response_format={"type": "json_object"}
                )
                content = response.choices[0].message.content

            return json.loads(content)
        except Exception as e:
            print(f"[LLMService] Inference error: {e}")
            return {"error": str(e)}

class LLM_LLAMA_LOCAL:
    def __init__(self, nodes):
        self.nodes = nodes

    def eval_prob(self, prompts, choices):
        # Stub implementation
        return [0.0] * len(choices)

    def _chat(self, prompt, history):
        # Stub implementation
        return "Ok"

class LLM_GPT_API:
    def __init__(self, models, api_key, org):
        self.service = LLMService(api_key=api_key, model=models[0] if models else "gpt-3.5-turbo")

    def __call__(self, args):
        # args is (res, hist)
        # Stub implementation returning empty dict as high() expects a dict/json
        return {}
