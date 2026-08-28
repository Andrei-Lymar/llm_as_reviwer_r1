import pandas as pd
import argparse
import sys
import os

for proxy_var in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
                  "http_proxy", "https_proxy", "all_proxy"]:
    os.environ.pop(proxy_var, None)

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI, APIError, APITimeoutError

API_KEY = 'sk-4e1282d63faa4f1fa659b714ce19e003' # Insert your key
BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-v4-flash"

# DeepSeek allow much more, but it is enough for 1k projects

MAX_WORKERS = 100

MAX_RETRIES = 3

# Autosave

SAVE_INTERVAL = 100

REQUEST_TIMEOUT = 120

THINKING_ENABLED = False

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# Row processing

def process_row(description, instruction):
    prompt = f"{instruction}\n\nОписание: {description}\n"

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0, # !!! temperature setup
    }
    if THINKING_ENABLED:
        payload["reasoning_effort"] = "high"
        payload["extra_body"] = {"thinking": {"type": "enabled"}}

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(**payload, timeout=REQUEST_TIMEOUT)
            return response.choices[0].message.content.strip()
        except (APITimeoutError, APIError) as e:
            if attempt < MAX_RETRIES - 1:
                continue
            return f"Error after {MAX_RETRIES} tries: {e}"
        except Exception as e:
            return f"Processing error: {e}"

def format_response(text):
    return text.replace('\n', ' ').strip() if text else ""

def load_or_init_data(input_file, output_file):
    try:
        df = pd.read_excel(output_file)
        if 'response' not in df.columns:
            df['response'] = ''
        unprocessed = len(df[df['response'].isna() | (df['response'] == '')])
        print(f"Save detected. Unprocessed rows: {unprocessed}")
        return df
    except FileNotFoundError:
        df = pd.read_excel(input_file)
        df['response'] = ''
        print("Launch a new processing")
        return df

# Main part
def main():
    parser = argparse.ArgumentParser(description='Projects evaluation by DeepSeek')
    parser.add_argument('input_file', help='Input Excel (.xlsx)')
    parser.add_argument('instruction_file', help='Prompt (.txt)')
    parser.add_argument('--workers', type=int, default=MAX_WORKERS, help=f'Concurrent requests (def. {MAX_WORKERS})')
    parser.add_argument('--output', default=None, help='File with results (def. output.xlsx in the script folder)')
    args = parser.parse_args()

    for path, label in [(args.input_file, 'Projects'), (args.instruction_file, 'Prompt')]:
        if not os.path.exists(path):
            print(f"Error: {label} '{path}' not found"); sys.exit(1)

    with open(args.instruction_file, 'r', encoding='utf-8') as f:
        instruction = f.read()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = args.output or os.path.join(script_dir, 'output.xlsx')

    print(f"Projects list:   {args.input_file}")
    print(f"Prompt:     {args.instruction_file}")
    print(f"Results will be in:  {output_file}")
    print(f"Model:         {MODEL}")
    print(f"Reasoning:    {'On' if THINKING_ENABLED else 'Off'}")
    print(f"Concurrency:    {args.workers} workers")
    print("-" * 50)

    df = load_or_init_data(args.input_file, output_file)

    todo = df[df['response'].isna() | (df['response'] == '')].index.tolist()
    if not todo:
        print("Everything is done!"); return
    print(f"To process: {len(todo)} rows")

    lock = threading.Lock()
    done = 0

    def work(index):
        description = ' | '.join(
            f"{col}: {df.at[index, col]}"
            for col in df.columns
            if col != 'response'
            and pd.notna(df.at[index, col])
            and str(df.at[index, col]).strip() != ''
        )
        return index, format_response(process_row(description, instruction))

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(work, idx): idx for idx in todo}
        try:
            for future in as_completed(futures):
                index, resp = future.result()
                with lock:
                    df.at[index, 'response'] = resp
                    done += 1
                    preview = resp[:80] + "..." if len(resp) > 80 else resp
                    print(f"[{done}/{len(todo)}] row {index+1}: {preview}")
                    if done % SAVE_INTERVAL == 0:
                        df.to_excel(output_file, index=False)
                        print(f"Autosave after ({done})")
        except KeyboardInterrupt:
            print("\n⚠️ Canceled. Saving...")

    df.to_excel(output_file, index=False)
    print("Done")

if __name__ == "__main__":
     main()
