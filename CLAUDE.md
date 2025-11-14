# CLAUDE.md - AI Assistant Guide for text-generation-webui

**Last Updated**: 2025-11-14
**Repository**: https://github.com/oobabooga/text-generation-webui
**Purpose**: Comprehensive guide for AI assistants working with this codebase

---

## Table of Contents

1. [Repository Overview](#repository-overview)
2. [Codebase Structure](#codebase-structure)
3. [Core Architecture](#core-architecture)
4. [Key Components](#key-components)
5. [Extension System](#extension-system)
6. [API Architecture](#api-architecture)
7. [Configuration Management](#configuration-management)
8. [Development Workflows](#development-workflows)
9. [Coding Conventions](#coding-conventions)
10. [Common Tasks](#common-tasks)
11. [Important Files Reference](#important-files-reference)
12. [Testing Guidelines](#testing-guidelines)
13. [Troubleshooting](#troubleshooting)

---

## Repository Overview

### What is text-generation-webui?

A Gradio-based web UI for Large Language Models with the goal of becoming the "AUTOMATIC1111/stable-diffusion-webui" of text generation.

**Key Features**:
- Multiple model loader backends: Transformers, llama.cpp, ExLlamaV3, ExLlamaV2, HQQ, TensorRT-LLM
- OpenAI-compatible API with Chat and Completions endpoints
- Automatic prompt formatting using Jinja2 templates
- Three chat modes: instruct, chat-instruct, and chat
- Extension system for plugins
- LoRA fine-tuning support
- Completely private (no telemetry)

**Tech Stack**:
- **Frontend**: Gradio 4.37, JavaScript, CSS
- **Backend**: Python 3.11, FastAPI (for API endpoints)
- **ML Frameworks**: PyTorch 2.6, Transformers 4.50, Accelerate 1.5
- **Package Manager**: Conda (in `installer_files/`)

---

## Codebase Structure

```
text-generation-webui/
├── server.py                       # Main entry point (starts UI + API)
├── download-model.py               # Model download utility
├── one_click.py                    # Installation automation
│
├── modules/                        # Core application logic (30+ modules)
│   ├── shared.py                  # Global state, settings, args parser
│   ├── models.py                  # Model loading/unloading orchestration
│   ├── loaders.py                 # Model loader registry & parameters
│   ├── chat.py                    # Chat logic, prompt formatting (Jinja2)
│   ├── text_generation.py         # Generation pipeline & streaming
│   ├── extensions.py              # Extension loading & hook system
│   ├── ui*.py                     # Gradio UI components (8 modules)
│   ├── *_loader.py                # Backend-specific loaders
│   ├── prompts.py                 # Prompt/template management
│   ├── presets.py                 # Sampling parameter presets
│   ├── training.py                # LoRA fine-tuning
│   ├── LoRA.py                    # LoRA adapter application
│   └── utils.py                   # File/model discovery utilities
│
├── extensions/                     # Plugin system (18 built-in)
│   ├── example/                   # Template extension
│   ├── openai/                    # OpenAI API server (FastAPI)
│   ├── superboogav2/              # RAG/vector search
│   ├── silero_tts/                # Text-to-speech
│   ├── whisper_stt/               # Speech-to-text
│   └── [13 others]/
│
├── user_data/                      # User configuration & data
│   ├── models/                    # Downloaded models
│   │   └── config.yaml           # Per-model settings (regex matching)
│   ├── instruction-templates/     # Chat format templates (20+)
│   ├── characters/                # Character definitions (JSON)
│   ├── prompts/                   # Prompt library
│   ├── presets/                   # Sampling parameter presets
│   ├── loras/                     # LoRA adapters
│   ├── training/                  # Training datasets
│   └── settings.yaml              # Global settings (auto-loaded)
│
├── requirements/                   # Python dependencies
│   ├── full/                      # Full feature set
│   └── portable/                  # Minimal dependencies
│
├── css/                            # Frontend styling
├── js/                             # Frontend JavaScript
├── docs/                           # Documentation
└── docker/                         # Container definitions

```

---

## Core Architecture

### Application Flow

```
[User] → [Gradio UI] → [modules/ui_*.py] → [modules/chat.py]
                                          → [modules/text_generation.py]
                                          → [Model Loader] → [LLM Backend]
                                          → [Extensions] (hooks at each stage)
                                          → [Response Streaming]
```

### State Management

**Global State** (`modules/shared.py`):
- `shared.model` - Currently loaded model
- `shared.tokenizer` - Model tokenizer
- `shared.gradio` - Dictionary of all Gradio UI elements
- `shared.settings` - Settings dictionary (40+ parameters)
- `shared.args` - Command-line arguments
- `shared.generation_lock` - Thread safety for concurrent requests

**Settings Priority** (highest to lowest):
1. Command-line arguments
2. `user_data/settings.yaml` (auto-loaded)
3. `--settings` flag path
4. Built-in defaults

### Model Loading Process

```
1. Parse model name/path
2. Check user_data/models/config.yaml for per-model settings
3. Auto-detect loader OR use --loader flag
4. Apply loader-specific parameters
5. Load model into memory (GPU/CPU)
6. Extract metadata (context size, special tokens)
7. Apply LoRA adapters if specified
8. Ready for generation
```

**Supported Loaders**:
- `llama.cpp` - GGUF format, CPU/GPU offloading
- `Transformers` - HuggingFace models, quantization support
- `ExLlamav3_HF` - Optimized inference framework
- `ExLlamav2_HF` / `ExLlamav2` - Enhanced variants
- `HQQ` - Half-precision quantization
- `TensorRT-LLM` - NVIDIA optimized

---

## Key Components

### 1. Chat System (`modules/chat.py`)

**Three Chat Modes**:

| Mode | Description | Use Case |
|------|-------------|----------|
| `instruct` | Single-turn instruction following | Task completion, Q&A |
| `chat` | Multi-turn conversation with character | Roleplay, assistants |
| `chat-instruct` | Hybrid: chat with instruction prefix | Conversational tasks |

**Template Processing** (Jinja2):
- Templates located in `user_data/instruction-templates/`
- Variables: `messages`, `name1`, `name2`, `user_bio`, `character_name`, etc.
- 20+ built-in formats: Mistral, Alpaca, Vicuna, ChatGLM, etc.

**History Management**:
- **Internal history**: Raw messages (saved to JSON)
- **Visible history**: Formatted for display
- Auto-save to `user_data/characters/{character}/` on changes
- Browse past conversations via UI

**Reference**: `modules/chat.py:1-800+`

### 2. Text Generation (`modules/text_generation.py`)

**Pipeline**:
```python
generate_reply()
  → _generate_reply()
  → apply_extensions('input_modifier')
  → shared.model.generate()
  → apply_stopping_strings()
  → apply_extensions('output_modifier')
  → return result
```

**Key Features**:
- Streaming output via callbacks
- Stopping strings (custom + built-in)
- Context window management
- Token counting
- Thread-safe generation lock

**Sampling Parameters** (30+ options):
- Temperature, top-p, top-k, top-a, min-p
- Repetition penalty, frequency penalty, presence penalty
- DynaTemp, Mirostat, Tail-Free Sampling
- Custom logits processors

**Reference**: `modules/text_generation.py:1-400+`

### 3. UI System (`modules/ui_*.py`)

**Modular Design**:

| Module | Tab | Key Components |
|--------|-----|----------------|
| `ui_chat.py` | Chat | Message display, character selector, history browser |
| `ui_default.py` | Default | Raw text input/output, markdown preview |
| `ui_notebook.py` | Notebook | Multi-segment generation |
| `ui_parameters.py` | Parameters | Sampling controls (sliders, dropdowns) |
| `ui_model_menu.py` | Model | Model loading, LoRA management |
| `ui_session.py` | Session | State export/import |

**Event Flow Pattern**:
```python
shared.gradio['Button'].click(
    gather_interface_values,      # Collect UI state
    gradio(input_elements),
    gradio('interface_state')
).then(
    processing_function,           # Process request
    gradio(inputs),
    gradio(outputs)
).then(
    update_ui,                     # Update display
    None, gradio(ui_elements)
)
```

**Reference**: `modules/ui_chat.py:1-500+`

### 4. Extension System (`modules/extensions.py`)

**Architecture**: Hook-based plugin system

**Discovery & Loading**:
```python
# Extensions located in extensions/{name}/
# Main file: extensions/{name}/script.py
importlib.import_module(f"extensions.{name}.script")
```

**Standard Hook Functions**:

```python
# Configuration
params = {}                        # Settings dict (saved to settings.yaml)

# Lifecycle
setup()                            # One-time initialization
ui()                               # Gradio component creation

# Generation hooks (in execution order)
history_modifier(history)          # Transform chat history
state_modifier(state)              # Modify generation state
chat_input_modifier(text, visible_text, state)  # User input
input_modifier(string, state)      # Prompt modification
bot_prefix_modifier(string)        # Bot response prefix
tokenizer_modifier(...)            # Token manipulation
logits_processor_modifier(list)    # Add LogitsProcessor
# [Generation happens]
output_modifier(string, state)     # Post-generation transform

# Custom functions
custom_generate_reply()            # Replace generation
custom_generate_chat_prompt()      # Replace prompt building

# UI
custom_css()                       # Inject CSS
custom_js()                        # Inject JavaScript
```

**Extension Structure**:
```
extension_name/
├── script.py           # Main module with hooks
├── requirements.txt    # Optional Python dependencies
├── README.md          # Documentation
└── [resources]/       # Assets, configs, etc.
```

**Built-in Extensions**:
- `openai` - OpenAI-compatible API (FastAPI)
- `superboogav2` - RAG with vector databases
- `silero_tts` - Text-to-speech synthesis
- `whisper_stt` - Speech-to-text transcription
- `sd_api_pictures` - Stable Diffusion integration
- `Training_PRO` - Advanced LoRA training
- `gallery`, `character_bias`, `long_replies`, etc.

**Reference**: `modules/extensions.py:1-200+`, `extensions/example/script.py`

---

## API Architecture

### OpenAI Extension (`extensions/openai/`)

**Framework**: FastAPI with uvicorn

**Authentication**: Bearer token via `--api-key` / `--admin-key`

**Key Endpoints**:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/v1/chat/completions` | POST | Chat completion (OpenAI-compatible) |
| `/v1/completions` | POST | Text completion (legacy) |
| `/v1/models` | GET | List available models |
| `/v1/embeddings` | POST | Generate embeddings |
| `/v1/internal/encode` | POST | Tokenize text |
| `/v1/internal/decode` | POST | Detokenize IDs |
| `/v1/internal/model/load` | POST | Load model (admin) |
| `/v1/internal/model/unload` | POST | Unload model (admin) |

**Starting the API**:
```bash
python server.py --api --api-key YOUR_KEY_HERE
# Or add to user_data/CMD_FLAGS.txt:
# --api
# --api-key YOUR_KEY_HERE
```

**Example Request**:
```python
import requests

response = requests.post(
    "http://localhost:5000/v1/chat/completions",
    headers={"Authorization": "Bearer YOUR_KEY_HERE"},
    json={
        "model": "model-name",
        "messages": [{"role": "user", "content": "Hello!"}],
        "temperature": 0.7,
        "stream": True
    }
)
```

**Reference**: `extensions/openai/script.py`, `extensions/openai/typing.py`

---

## Configuration Management

### 1. Global Settings (`user_data/settings.yaml`)

**Key Settings**:
```yaml
# Generation
max_new_tokens: 512
temperature: 0.7
top_p: 0.9
do_sample: true

# Chat
mode: chat
instruction_template: Alpaca
character: Assistant

# Display
chat_style: cai-chat
start_with: greeting

# Templates
custom_system_message: ""
user_bio: ""
```

**60+ available settings** covering display, generation, chat, templates.

### 2. Per-Model Settings (`user_data/models/config.yaml`)

**Pattern Matching** (case-insensitive regex):
```yaml
# Match any model with "mistral" in name
.*mistral:
  instruction_template: 'Mistral'
  loader: 'llama.cpp'
  n_ctx: 8192
  n_gpu_layers: 32

# Match specific model
llama-2-13b-chat:
  instruction_template: 'Llama-v2'
  skip_special_tokens: false
```

**Available Settings**:
- Loader selection and parameters
- Context size (`n_ctx`, `ctx_size`)
- GPU layers (`n_gpu_layers`)
- Instruction template
- Special token handling
- Quantization settings

### 3. Instruction Templates (`user_data/instruction-templates/`)

**Format** (YAML with Jinja2):
```yaml
instruction_template: |-
  {%- for message in messages %}
    {%- if message['role'] == 'system' -%}
      {{ message['content'] }}
    {%- elif message['role'] == 'user' -%}
      ### Instruction:
      {{ message['content'] }}
    {%- elif message['role'] == 'assistant' -%}
      ### Response:
      {{ message['content'] }}
    {%- endif %}
  {%- endfor %}
  {%- if add_generation_prompt %}
    ### Response:
  {%- endif %}
```

**20+ Built-in Templates**: Alpaca, Mistral, Vicuna, ChatGLM, Command-R, Llama-3, etc.

### 4. Command-Line Flags

**Configuration Priority**:
1. CLI flags (highest)
2. `user_data/CMD_FLAGS.txt` (one flag per line)
3. Settings files
4. Defaults

**Common Flags**:
```bash
--model MODEL_NAME              # Load model on startup
--loader llama.cpp              # Force specific loader
--n-gpu-layers 32               # GPU offloading (llama.cpp)
--api                           # Enable API
--listen                        # Accept network connections
--extensions openai superboogav2  # Load extensions
```

**Reference**: `python server.py --help` or README.md:184-319

---

## Development Workflows

### Setting Up Development Environment

```bash
# 1. Clone repository
git clone https://github.com/oobabooga/text-generation-webui
cd text-generation-webui

# 2. Create conda environment
conda create -n textgen python=3.11
conda activate textgen

# 3. Install PyTorch (NVIDIA GPU example)
pip3 install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124

# 4. Install requirements
pip install -r requirements/full/requirements.txt

# 5. Run server
python server.py
```

### Working with Extensions

**Creating a New Extension**:

1. Copy template:
```bash
cp -r extensions/example extensions/my_extension
```

2. Edit `extensions/my_extension/script.py`:
```python
params = {
    'enable': True,
    'custom_param': 'value'
}

def setup():
    """Initialize extension (called once)"""
    pass

def ui():
    """Create Gradio UI components"""
    import gradio as gr
    with gr.Column():
        gr.Markdown("# My Extension")
        # Add your UI here

def input_modifier(string, state):
    """Modify the prompt before generation"""
    # Modify string here
    return string

def output_modifier(string, state):
    """Modify the output after generation"""
    # Modify string here
    return string
```

3. Load extension:
```bash
python server.py --extensions my_extension
```

**Extension Best Practices**:
- Check `if params['enable']:` before processing
- Use logging: `from modules.logging_colors import logger`
- Handle errors gracefully (try/except)
- Document parameters in docstrings
- Add requirements.txt for dependencies

### Modifying Core Functionality

**Common Modification Points**:

| Task | File | Function |
|------|------|----------|
| Change generation logic | `modules/text_generation.py` | `generate_reply_*()` |
| Modify chat formatting | `modules/chat.py` | `generate_chat_prompt()` |
| Add UI elements | `modules/ui_*.py` | Tab-specific functions |
| Add model loader | `modules/loaders.py` | Add to `loaders_and_params` |
| Change stopping behavior | `modules/text_generation.py` | `apply_stopping_strings()` |

**Example: Adding a Custom Stopping String**:
```python
# In modules/text_generation.py
def apply_stopping_strings(reply, stopping_strings):
    # Add your custom logic
    if "CUSTOM_STOP" in reply:
        reply = reply.split("CUSTOM_STOP")[0]
    # ... existing logic
    return reply
```

### Testing Changes

**Manual Testing**:
1. Start server: `python server.py --verbose`
2. Test in UI: http://localhost:7860
3. Check logs in terminal

**API Testing**:
```bash
# Start with API enabled
python server.py --api --nowebui

# Test endpoint
curl http://localhost:5000/v1/models
```

**Extension Testing**:
```bash
# Load specific extensions
python server.py --extensions my_extension --verbose

# Check extension loaded
# Look for "Loading extensions..." in terminal
```

---

## Coding Conventions

### Python Style

**Naming**:
- `snake_case` for functions, variables, modules
- `PascalCase` for classes
- `UPPER_CASE` for constants
- Descriptive names (avoid abbreviations)

**Imports**:
```python
# Standard library
import os
import sys
from pathlib import Path

# Third-party
import torch
import gradio as gr
from transformers import AutoModelForCausalLM

# Local modules
from modules import shared
from modules.logging_colors import logger
```

**Error Handling**:
```python
try:
    result = risky_operation()
except Exception as e:
    logger.error(f"Operation failed: {e}")
    # Graceful degradation
    result = default_value
```

**Logging**:
```python
from modules.logging_colors import logger

logger.info("Informational message")
logger.warning("Warning message")
logger.error("Error message")
```

### Gradio Patterns

**Storing UI Elements**:
```python
# Add to shared.gradio dict with descriptive key
shared.gradio['my_button'] = gr.Button("Click me")
shared.gradio['my_textbox'] = gr.Textbox(label="Input")
```

**Event Chains**:
```python
# Sequential operations with .then()
shared.gradio['button'].click(
    function1, inputs, outputs1
).then(
    function2, inputs, outputs2
).then(
    function3, None, outputs3
)
```

**State Management**:
```python
# Backend-only state (not visible in UI)
state = gr.State({})

# Accessing state in functions
def my_function(input_text, state):
    state['last_input'] = input_text
    return state

# Wire it up
button.click(my_function, [textbox, state], state)
```

### Shared State Access

**Reading Shared State**:
```python
from modules import shared

current_model = shared.model
current_settings = shared.settings
```

**Modifying Shared State**:
```python
# Settings
shared.settings['temperature'] = 0.8

# Args (read-only, set at startup)
n_gpu_layers = shared.args.n_gpu_layers
```

**Thread Safety**:
```python
# Use generation lock for thread safety
with shared.generation_lock:
    result = shared.model.generate(...)
```

### Extension Patterns

**Parameter Definition**:
```python
params = {
    'enable': True,
    'param1': 'default_value',
    'param2': 100
}
```

**Hook Implementation**:
```python
def input_modifier(string, state):
    """
    Modifies the input prompt before generation.

    Args:
        string: The prompt text
        state: Interface state dict

    Returns:
        Modified prompt string
    """
    if not params['enable']:
        return string

    # Your modification logic
    modified = string.replace('X', 'Y')
    return modified
```

**Optional Hooks**:
```python
# Only implement hooks you need
# Missing hooks are safely ignored
```

---

## Common Tasks

### Adding a New Chat Template

1. Create template file:
```bash
# user_data/instruction-templates/MyTemplate.yaml
```

2. Define template:
```yaml
instruction_template: |-
  {%- for message in messages %}
    {%- if message['role'] == 'user' -%}
      User: {{ message['content'] }}
    {%- elif message['role'] == 'assistant' -%}
      Assistant: {{ message['content'] }}
    {%- endif %}
  {%- endfor %}
  {%- if add_generation_prompt %}
    Assistant:
  {%- endif %}
```

3. Use in UI or config:
```yaml
# user_data/settings.yaml
instruction_template: MyTemplate
```

### Loading Models Programmatically

```python
from modules import shared, models

# Unload current model
models.unload_model()

# Load new model
model_name = "llama-2-7b-chat.Q4_K_M.gguf"
shared.model_name = model_name
models.load_model(model_name)
```

### Accessing Model Generation

```python
from modules import text_generation, shared

# Prepare generation state
state = {
    'max_new_tokens': 100,
    'temperature': 0.7,
    'top_p': 0.9,
    'do_sample': True
}

# Generate
prompt = "Once upon a time"
for reply in text_generation.generate_reply(prompt, state, False):
    output = reply

print(output)
```

### Creating Custom Logits Processor

```python
from transformers import LogitsProcessor

class CustomLogitsProcessor(LogitsProcessor):
    def __init__(self, custom_param):
        self.custom_param = custom_param

    def __call__(self, input_ids, scores):
        # Modify scores (token probabilities)
        # scores shape: [batch_size, vocab_size]

        # Example: Boost certain token IDs
        token_ids = [123, 456, 789]
        scores[:, token_ids] += 2.0

        return scores

# Use in extension
def logits_processor_modifier(processor_list, input_ids):
    processor = CustomLogitsProcessor(custom_param=5)
    processor_list.append(processor)
```

### Downloading Models from Code

```python
import subprocess

# Using download-model.py
model_name = "TheBloke/Llama-2-7B-Chat-GGUF"
subprocess.run([
    "python", "download-model.py",
    model_name,
    "--specific-file", "llama-2-7b-chat.Q4_K_M.gguf"
])
```

### Adding Custom Sampling Parameters

1. Add to `modules/loaders.py`:
```python
# In loaders_and_params dict
'Transformers': [
    # ... existing params
    'my_custom_param'
]
```

2. Use in generation:
```python
# modules/transformers_loader.py or extension
def my_generate_function(prompt, state):
    custom_value = state.get('my_custom_param', default_value)
    # Use in generation
```

3. Add UI control:
```python
# In modules/ui_parameters.py or extension ui()
shared.gradio['my_custom_param'] = gr.Slider(
    minimum=0, maximum=10, value=5,
    label='My Custom Parameter'
)
```

---

## Important Files Reference

### Core Files (Modify with Caution)

| File | Purpose | Modification Risk |
|------|---------|------------------|
| `server.py` | Entry point, Gradio app setup | High |
| `modules/shared.py` | Global state, settings | High |
| `modules/models.py` | Model loading logic | High |
| `modules/text_generation.py` | Generation pipeline | Medium-High |
| `modules/chat.py` | Chat formatting | Medium |
| `modules/extensions.py` | Extension system | High |
| `modules/ui_*.py` | UI components | Medium |

### Safe to Modify

| Location | Purpose | Notes |
|----------|---------|-------|
| `user_data/instruction-templates/` | Chat templates | Safe, user config |
| `user_data/settings.yaml` | Global settings | Safe, user config |
| `user_data/models/config.yaml` | Per-model config | Safe, user config |
| `extensions/*/` | Extension code | Safe, isolated |
| `css/main.css` | Styling | Safe, cosmetic |
| `js/main.js` | Frontend behavior | Medium, affects UX |

### Configuration Files

| File | Format | Purpose |
|------|--------|---------|
| `user_data/settings.yaml` | YAML | Global UI/generation settings |
| `user_data/models/config.yaml` | YAML | Per-model loader & params |
| `user_data/CMD_FLAGS.txt` | Text | Command-line flags (one per line) |
| `requirements/full/requirements.txt` | pip | Python dependencies |

### Documentation

| File | Content |
|------|---------|
| `README.md` | Installation, features, CLI flags |
| `docs/*.md` | Feature-specific documentation |
| `extensions/*/README.md` | Extension-specific docs |

---

## Testing Guidelines

### Current State

- **Limited formal testing**: No dedicated test suite
- **Manual testing**: Primary validation method
- **Logging**: Verbose mode for debugging

### Testing Checklist for Changes

**Before Committing**:
1. Test with `--verbose` flag to see detailed logs
2. Test with at least one model loader (preferably llama.cpp for speed)
3. Verify UI still loads without errors
4. Check console for Python exceptions
5. Test both chat and default modes
6. If modifying API: test with curl/Postman

**For Extension Changes**:
1. Load extension with `--extensions your_extension`
2. Check extension params saved to settings.yaml
3. Verify hooks execute (add logging)
4. Test with extension disabled (params['enable'] = False)

**For UI Changes**:
1. Test in both light and dark themes (if applicable)
2. Check responsive layout at different window sizes
3. Verify Gradio events fire correctly
4. Test keyboard shortcuts still work

### Running Verbose Mode

```bash
# See all prompts and generation details
python server.py --verbose --extensions my_extension

# See API request/response
python server.py --api --verbose
```

### Debugging Tips

**Enable Detailed Logging**:
```python
from modules.logging_colors import logger
import logging

logger.setLevel(logging.DEBUG)
logger.debug("Detailed debug message")
```

**Check Gradio State**:
```python
# Add to function
print(f"State: {state}")
print(f"Shared settings: {shared.settings}")
```

**Test Extension Hooks**:
```python
# In extension script.py
def input_modifier(string, state):
    print(f"[MY_EXT] Input: {string}")
    # Your logic
    print(f"[MY_EXT] Output: {modified_string}")
    return modified_string
```

---

## Troubleshooting

### Common Issues

**Model Not Loading**:
- Check `user_data/models/` contains model files
- Verify loader auto-detection or use `--loader` flag
- Check RAM/VRAM availability
- Review logs for specific errors

**API Not Accessible**:
- Add `--api` flag or to `CMD_FLAGS.txt`
- Check firewall (port 5000 default)
- Use `--listen` for network access
- Verify `--api-key` if authentication enabled

**Extension Not Loading**:
- Check extension name in `--extensions` flag
- Verify `script.py` exists in `extensions/{name}/`
- Install extension requirements: `pip install -r extensions/{name}/requirements.txt`
- Check console for import errors

**Generation Hangs**:
- Check `shared.generation_lock` not deadlocked
- Verify model loaded: `shared.model is not None`
- Check context size vs. prompt length
- Review stopping strings configuration

**UI Not Updating**:
- Verify Gradio event chains connected
- Check function returns match output components
- Use `.then()` for sequential updates
- Inspect browser console for JavaScript errors

### Performance Issues

**Slow Generation**:
- Increase GPU layers: `--n-gpu-layers 32` (llama.cpp)
- Use quantized models (Q4_K_M, Q5_K_M)
- Enable flash attention: `--flash-attn`
- Reduce context size: `--ctx-size 2048`

**High Memory Usage**:
- Reduce GPU layers (offload to CPU)
- Use smaller model
- Enable memory-efficient attention
- Clear KV cache between generations

**Gradio Lag**:
- Reduce streaming token rate
- Simplify custom CSS/JS
- Disable verbose logging
- Use `--queue` mode for better concurrency

### Git Workflow Issues

**Branch Naming**:
- Always create branches prefixed with `claude/`
- Include session ID suffix for auto-cleanup
- Example: `claude/add-feature-abc123xyz`

**Push Failures**:
- Verify branch name format
- Check network connectivity
- Retry with exponential backoff (up to 4 times)
- Ensure git credentials configured

---

## Best Practices for AI Assistants

### When Working with This Codebase

1. **Understand Before Modifying**:
   - Read relevant files first (`Read` tool)
   - Check existing patterns in similar code
   - Review documentation in `docs/`

2. **Prefer Extensions Over Core Changes**:
   - Create extension for new features when possible
   - Modify core only when necessary
   - Extensions are isolated and safer

3. **Maintain Backward Compatibility**:
   - Don't break existing APIs
   - Keep default behavior unchanged
   - Add new parameters as optional

4. **Test Thoroughly**:
   - Use `--verbose` flag
   - Test with real models
   - Verify UI still works
   - Check API endpoints if modified

5. **Follow Conventions**:
   - Use existing naming patterns
   - Match indentation style
   - Add logging for important operations
   - Document complex logic

6. **Communicate Changes**:
   - Clear commit messages
   - Update documentation if needed
   - Note breaking changes
   - Explain rationale in comments

### Code Review Checklist

Before submitting changes:
- [ ] Code follows existing style conventions
- [ ] No breaking changes to public APIs
- [ ] Logging added for debugging
- [ ] Error handling implemented
- [ ] Tested manually with `--verbose`
- [ ] Documentation updated if needed
- [ ] Settings saved properly (if config added)
- [ ] UI elements added to `shared.gradio` dict
- [ ] Extension hooks return correct types
- [ ] Thread safety considered for shared state

---

## Additional Resources

### Official Documentation
- Wiki: https://github.com/oobabooga/text-generation-webui/wiki
- API Examples: https://github.com/oobabooga/text-generation-webui/wiki/12-OpenAI-API#examples
- Extensions: https://github.com/oobabooga/text-generation-webui/wiki/07-Extensions

### Community
- Reddit: https://www.reddit.com/r/Oobabooga/
- Issues: https://github.com/oobabooga/text-generation-webui/issues

### Related Projects
- Extension Directory: https://github.com/oobabooga/text-generation-webui-extensions
- Docker Images: https://github.com/Atinoda/text-generation-webui-docker

---

## Revision History

- **2025-11-14**: Initial creation - Comprehensive codebase analysis and documentation

---

**Note**: This document is maintained for AI assistants working with the text-generation-webui codebase. Keep it updated as the project evolves.
