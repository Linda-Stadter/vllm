# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import os

os.environ["VLLM_USE_V1"] = "0"

from huggingface_hub import snapshot_download

from vllm import LLM, SamplingParams

model = 'ai4bharat/indictrans2-en-indic-1B'
if not os.path.exists(model):
    print(f"Downloading model {model}...")
    snapshot_download(repo_id=model,
                      local_dir=model,
                      local_dir_use_symlinks=False)

config_path = os.path.join(model, "config.json")
if os.path.exists(config_path):
    with open(config_path) as f:
        config = json.load(f)

    if 'd_model' not in config:
        config['d_model'] = config.get('decoder_embed_dim', 1024)
        print(f"Added d_model={config['d_model']} to config.json")

        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

llm = LLM(model=model, trust_remote_code=True, enforce_eager=True)

sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=100,
)

# Test prompts
test_prompts = [
    "eng_Latn hin_Deva When I was young, I used to go to the park every day.",
    "eng_Latn hin_Deva We watched a new movie last week, which was very inspiring.",
    "eng_Latn hin_Deva If you had met me at that time, we would have gone out to eat.",
    "eng_Latn hin_Deva My friend has invited me to his birthday party, and I will give him a gift.",
]

outputs = llm.generate(test_prompts, sampling_params=sampling_params)

for output in outputs:
    generated_text = output.outputs[0].text
    generated_token_ids = output.outputs[0].token_ids
    print(f"Output: {generated_text}")
    print(f"Token IDs: {generated_token_ids}")
