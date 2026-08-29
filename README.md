# Omni-UBench

The official repository of "Omni-UBench: Benchmarking Omni-Modal LLM-based Guardrails Across Safety, Utility, and Reliability".
**Omni-UBench** is a comprehensive framework for systematically evaluating omni-modal guardrails across *safety, utility and reliability*.

> [!CAUTION]
> Harmful Content Disclaimer:
> This project involves safety benchmarks that may contain harmful, disturbing, or offensive examples.
> These data are provided strictly for safety research and guardrail development.

## Quick Start

### Environment Setup [Required]

This framework is built on Python 3.10, PyTorch 2.10, Transformers >= 5.2, and vLLM. Create and install the recommended environment with one script:

```bash
# Prerequisite: conda installation
bash env_setup.sh

# Optional: disable flash-attn install
# INSTALL_FLASH_ATTN=0 bash env_setup.sh

# Optional: manually install dependencies in an existing Python 3.10 env
# pip install -r requirements.txt
```

### Benchmark Preparation [Required]

The full **Omni-UBench** consists of 183K+ samples collected from 35+ publicly available datasets, covering diverse modalities, risk categories, and safety evaluation scenarios.

Due to the repository size limitation (50 MB), we only include a **preview (Mini) version of Omni-UBench** in this path (data/omniubench.zip). Unzip it before inference.

```bash
unzip data/omniubench.zip -d data/omniubench
```

The mini version provides a randomly sampled subset of the benchmark, enabling users to quickly test the evaluation pipeline. (Complete benchmark will be released soon!)

Meanwhile, we provide a step-by-step dataset construction pipeline to facilitate full benchmark reproduction. By downloading the corresponding publicly available datasets and executing the provided scripts, users can automatically build Omni-UBench, and use it with our provided library modules.

### Prepare OLLMs/Guardrails [Required]

Execute `prepare_base_models_hf.sh`/`prepare_guardrails_hf.sh` to download omni foundation models/specified guardrails. feel free to uncomment other model options in them.

```bash
bash prepare_base_models_hf.sh
# or/and
bash prepare_guardrails_hf.sh
```

### A Quick Inference Example

The script loads a base Omni model (Qwen2.5-Omni-7B) and runs some test cases based on the mini-version of Omni-UBench. By default the script filter queries from a data source (e.g., TruthfulQA). Be free to check and modify the data source to be evaluated!

```bash
python inference_example.py

# if use specified CUDA device (vllm)
CUDA_VISIBLE_DEVICES=0 python inference_example.py
```

## Prepararing All Data Sources [Optional]

Most benchmark datasets are publicly accessible from HuggingFace and can be prepared directly with our one-click script `prepare_omniubench_hf.sh`. Some datasets requiring alternative fetching (e.g., `git clone`). See `hf_datasets.txt` for the dataset list.

First, execute our one-click script `prepare_omniubench_hf.sh`.

```bash
# Recommend to login to Hugging Face and generate an access token, since some datasets require confirmation and access permissions.
hf auth login

# By default downloaded under data/...
bash prepare_omniubench_hf.sh
```

Then, fetch data from github repo, and extracts labels into the data directory.

```bash
# Prerequisite: git
bash prepare_omniubench_external.sh
```

### Data Sources

Omni-UBench is constructed by integrating publicly available safety and evaluation datasets across different modalities. The following table lists all benchmark sources supported by Omni-UBench.

| Modality | Source            | Repo ID                                                    |
| :------: | :---------------- | :--------------------------------------------------------- |
|   Text   | MT-Bench          | HuggingFace:`HuggingFaceH4/mt_bench_prompts`             |
|   Text   | TruthfulQA        | HuggingFace:`domenicrosati/TruthfulQA`                   |
|   Text   | AdvBench          | HuggingFace:`walledai/AdvBench`                          |
|   Text   | DAN               | HuggingFace:`verazuo/jailbreak-llms`                     |
|   Text   | JailbreakBench    | GitHub:`JailbreakBench/jailbreakbench`                   |
|   Text   | CipherChat        | GitHub:`RobustNLP/CipherChat`                            |
|   Text   | HarmBench         | HuggingFace:`walledai/HarmBench`                         |
|   Text   | SimpleSafetyTests | HuggingFace:`Bertievidgen/SimpleSafetyTests`             |
|   Text   | XSTest            | HuggingFace:`Paul/XSTest`                                |
|   Text   | OpenAI Moderation | GitHub:`openai/moderation-api-release`                   |
|   Text   | ToxicChat         | HuggingFace:`lmsys/toxic-chat`                           |
|   Text   | WildGuard         | HuggingFace:`allenai/wildguardmix`                       |
|   Text   | Aegis2.0          | HuggingFace:`nvidia/Aegis-AI-Content-Safety-Dataset-2.0` |
|   Text   | Beavertails       | HuggingFace:`PKU-Alignment/BeaverTails`                  |
|   Text   | SafeRLHF          | HuggingFace:`PKU-Alignment/PKU-SafeRLHF-QA`              |
|  Image  | OK-VQA            | HuggingFace:`lmms-lab/OK-VQA`                            |
|  Image  | MME               | HuggingFace:`lmms-lab/MME`                               |
|  Image  | llava-wild        | HuggingFace:`lmms-lab/llava-bench-in-the-wild`           |
|  Image  | MM-Vet            | GitHub:`yuweihao/MM-Vet`                                 |
|  Image  | MM-SafetyBench    | HuggingFace:`PKU-Alignment/MM-SafetyBench`               |
|  Image  | FigStep           | GitHub:`CryptoAILab/FigStep`                             |
|  Image  | JailbreakV-28K    | HuggingFace:`JailbreakV-28K/JailBreakV-28k`              |
|  Image  | VLSafe            | HuggingFace:`YangyiYY/LVLM_NLF`                          |
|  Image  | Hades             | HuggingFace:`Monosail/HADES`                             |
|  Image  | VLSBench          | HuggingFace:`Foreshhh/vlsbench`                          |
|  Image  | MML-Safebench     | GitHub:`wangyu-ovo/MML`                                  |
|  Image  | RTVLM             | HuggingFace:`MMInstruction/RedTeamingVLM`                |
|  Image  | VLGuard           | GitHub:`ys-zong/VLGuard`                                 |
|  Image  | SIUO              | HuggingFace:`sinwang/SIUO`                               |
|  Audio  | AIAH              | GitHub:`YangHao97/RedteamAudioLMMs`                      |
|  Audio  | AJailBench        | HuggingFace:`MBZUAI/AudioJailbreak`                      |
|  Audio  | VoiceBench        | HuggingFace:`hlt-lab/voicebench`                         |
|  Video  | MMBench-Video     | HuggingFace:`opencompass/MMBench-Video`                  |
|  Video  | Video-SafetyBench | HuggingFace:`BAAI/Video-SafetyBench`                     |
|  Video  | SafeWatch         | HuggingFace:`Zhaorun/SafeWatch-Bench`                    |
|   Omni   | Safebench         | HuggingFace:`Zonghao2025/safebench`                      |
|   Omni   | Omni-SafetyBench  | HuggingFace:`Leyiii/Omni-SafetyBench`                    |

User can access and download them by link to huggingface/github websites: https://huggingface.co/{repo_id} or https://github.com/{repo_id}.

## Batch Benchmarking

This step loads your base model and runs benchmarking with batched subsets.

```bash
# Example: run mini benchmark on GPU 0
CUDA_VISIBLE_DEVICES=0 python eval_omniguard_sft_all.py \
  --model_path ./models/base/Qwen2.5-Omni-7B \
  --enable_vllm
# If custom data_dir is specified
DATA_DIR=/path/to/data CUDA_VISIBLE_DEVICES=0 python eval_omniguard_sft_all.py --model_path ./models/base/Qwen2.5-Omni-7B --enable_vllm
```

Omni-UBench adopts a multi-dimensional evaluation protocol (Safety-Utility-Reliability),
and datasets are grouped accordingly for target-oriented evaluation:

- safety: includes basic safety and jailbreak defense (detecting malicious queries and jailbreak samples)
- utility: evaluates recognition/interference on normal and safe samples
- reliability: evaluates safety hallicination, spurious correlations (False rejection to jailbreak-like textual prompts), and cross-modal modality bias.

Be free to uncomment the benchmark settings in `eval_omniguard_sft_all.py` for a full benchmarking!

```python
eval_sets = {
  # For fast evaluate the whole inference framework
  "fast": eval_fast_validate,

  # Uncomment for complete evaluation
  "utility": eval_general,
  "safety": eval_basic_safety,
  "jailbk": eval_ext_jailbreaks,

  "fr": eval_false_reject,
  "gh": eval_mask_all_exp,
}
```

Expected outputs:

- Displayed Metrics for each dataset/benchmark
- Per-dataset predictions under `output/metric.logs/<model_namh e>.<prompt-version>/`
- Brief log under `logs/<model_name>.<version>.metric.log`

## Inference other Baselines

- GuardReasoner-Omni

```bash
CUDA_VISIBLE_DEVICES=0 python -m baseline.bsl_guardreasoner --dataset text.advbench
```

- OmniGuard

```bash
python -m baseline.bsl_omniguard --dataset text.advbench
```

The above dataset name is formatted as 'modality'.'data_source_key', modality should be one of 'text/image/audio/video/omni', data_source_key is the field name of data-source config defined in `module/predefined_modality.py`. See it for more details.

- Gemini/MiMo

```bash
# Mimo V2.5 For example
python -m baseline.bsl_openai --batch_config data/infer.json --api_key <your-api-key> --base_url https://api.xiaomimimo.com/v1
```

The above config file `data/infer.json` defines the evaluated sources of Omni-UBench, and it can be modified accordingly. See `baseline/bsl_openai.py` and `baseline/bsl_openrouter.py` for more details.
