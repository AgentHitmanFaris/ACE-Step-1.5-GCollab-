import gradio as gr
import requests
import json
import time
import os
import sys
import webbrowser
from urllib.parse import urlencode

# The URL of your Colab notebook
COLAB_NOTEBOOK_URL = "https://colab.research.google.com/github/AgentHitmanFaris/ACE-Step-1.5-GCollab-/blob/main/ACE_Step_Colab.ipynb"

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from acestep.env_utils import check_environment

def get_audio_filename(url):
    import urllib.parse
    parsed_url = urllib.parse.urlparse(url)
    params = urllib.parse.parse_qs(parsed_url.query)
    path = params.get('path', [''])[0]
    return os.path.basename(path)

# Config persistence
CONFIG_FILE = os.path.join(PROJECT_ROOT, ".thin_client_config.json")

def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
        except:
            pass
    return {"url": "", "key": ""}

def save_config(url, key):
    with open(CONFIG_FILE, 'w') as f:
        json.dump({"url": url, "key": key}, f)

class ColabClient:
    def __init__(self):
        self.api_url = ""
        self.api_key = ""

    def set_config(self, url, key):
        self.api_url = url.rstrip('/')
        self.api_key = key
        save_config(self.api_url, self.api_key)
        try:
            resp = requests.get(f"{self.api_url}/health", timeout=5)
            if resp.status_code == 200:
                return "Successfully connected to Colab API!"
            else:
                return f"Connection failed (Status {resp.status_code})"
        except Exception as e:
            return f"Error: {str(e)}"

    def generate(self, prompt, lyrics, duration, steps, cfg, seed, thinking, language):
        if not self.api_url:
            return None, "Please set Colab URL first", ""

        payload = {
            "prompt": prompt,
            "lyrics": lyrics,
            "audio_duration": duration,
            "inference_steps": steps,
            "guidance_scale": cfg,
            "seed": seed,
            "use_random_seed": seed == -1,
            "thinking": thinking,
            "vocal_language": language,
            "ai_token": self.api_key
        }

        try:
            # 1. Release Task
            resp = requests.post(f"{self.api_url}/release_task", json=payload)
            if resp.status_code != 200:
                return None, f"Error: {resp.text}", ""
            
            data = resp.json()
            if data['code'] != 200:
                return None, f"API Error: {data['error']}", ""
            
            task_id = data['data']['task_id']
            yield None, "Generation in progress...", f"Task ID: {task_id}"

            # 2. Query Result
            # Note: The api_routes.py implementation of release_task is synchronous 
            # (it returns only after generation completes).
            query_payload = {
                "task_id_list": [task_id],
                "ai_token": self.api_key
            }
            resp = requests.post(f"{self.api_url}/query_result", json=query_payload)
            
            if resp.status_code == 200:
                data = resp.json()
                if data['code'] == 200 and data['data']:
                    res = data['data'][0]
                    if res['status'] == 1:
                        results = json.loads(res['result'])
                        if results:
                            # Use the first audio result
                            item = results[0]
                            audio_url = f"{self.api_url}{item['url']}"
                            if self.api_key:
                                audio_url += f"&ai_token={self.api_key}"
                            
                            yield audio_url, "Generation successful!", f"Task ID: {task_id}"
                            return
            
            yield None, "Generation failed or timed out", ""

        except Exception as e:
            yield None, f"Error: {str(e)}", ""

client = ColabClient()

def open_colab():
    webbrowser.open(COLAB_NOTEBOOK_URL)
    return "Opening Colab in your browser... Please click 'Runtime' -> 'Run all' there."

def create_ui():
    cfg = load_config()
    with gr.Blocks(title="ACE-Step Colab Thin Client") as demo:
        gr.Markdown("# ACE-Step Colab Thin Client")
        gr.Markdown("Run the UI locally, generate on Google Colab.")

        with gr.Row():
            with gr.Column(scale=2):
                open_btn = gr.Button("🚀 1. Open Colab Notebook", variant="secondary")
                open_msg = gr.Markdown("")
            with gr.Column(scale=3):
                colab_url = gr.Textbox(label="2. Paste Colab URL here", placeholder="https://xxxx.gradio.live", value=cfg["url"])
                api_key = gr.Textbox(label="API Key (Optional)", type="password", value=cfg["key"])
                connect_btn = gr.Button("3. Connect", variant="primary")
        
        status_msg = gr.Markdown("*Not connected*")

        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(label="Prompt", placeholder="Describe the music style...", lines=3)
                lyrics = gr.Textbox(label="Lyrics", placeholder="[inst] for instrumental", lines=5)
                
                with gr.Row():
                    duration = gr.Slider(minimum=10, maximum=600, value=30, step=10, label="Duration (sec)")
                    steps = gr.Slider(minimum=1, maximum=100, value=8, step=1, label="Inference Steps")
                
                with gr.Row():
                    cfg = gr.Slider(minimum=1.0, maximum=20.0, value=7.0, step=0.5, label="Guidance Scale")
                    seed = gr.Number(label="Seed (-1 for random)", value=-1, precision=0)
                
                thinking = gr.Checkbox(label="Enable LM Thinking", value=False)
                language = gr.Dropdown(choices=["en", "zh", "ja", "ko", "es", "fr", "de"], value="en", label="Vocal Language")
                
                generate_btn = gr.Button("Generate Music", variant="primary")

            with gr.Column():
                output_audio = gr.Audio(label="Generated result")
                output_log = gr.Textbox(label="Log", interactive=False)
                task_status = gr.Label(label="Task Status")

        # Events
        open_btn.click(open_colab, outputs=open_msg)
        connect_btn.click(client.set_config, inputs=[colab_url, api_key], outputs=status_msg)
        
        generate_btn.click(
            client.generate, 
            inputs=[prompt, lyrics, duration, steps, cfg, seed, thinking, language],
            outputs=[output_audio, output_log, task_status]
        )

    return demo

if __name__ == "__main__":
    # Ensure we are in the right environment
    check_environment()
    
    # Automatically open Colab on startup
    print(f"Opening Colab notebook: {COLAB_NOTEBOOK_URL}")
    webbrowser.open(COLAB_NOTEBOOK_URL)
    
    demo = create_ui()
    demo.launch(server_name="127.0.0.1", server_port=7861)
