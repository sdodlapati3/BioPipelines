#!/bin/bash
# ============================================================================
# BioPipelines LLM Provider API Keys
# ============================================================================
# Add these to your ~/.bashrc or source this file before running BioPipelines
#
# FREE TIER PROVIDERS (recommended - add these first!):
# ============================================================================

# 1. Google AI Studio (Gemini) - Already set ✅
#    Get key: https://aistudio.google.com/apikey
#    Limits: 250 req/day, 1M tokens/day
# export GOOGLE_API_KEY="your_key_here"

# 2. Cerebras - MOST GENEROUS FREE TIER! 🌟
#    Get key: https://cloud.cerebras.ai/
#    Limits: 14,400 req/day, 1M tokens/day
#    Models: Llama 3.3 70B, Qwen3 235B (free!), gpt-oss-120b
# export CEREBRAS_API_KEY="csk-your-key-here"  # <-- ADD YOUR KEY HERE (starts with csk-)

# 3. Groq - FASTEST INFERENCE! ⚡
#    Get key: https://console.groq.com/keys
#    Limits: 1,000 req/day (70B), 14,400 req/day (8B)
#    Models: Llama 3.3 70B, gpt-oss-120b
# export GROQ_API_KEY="gsk_your-key-here"  # <-- ADD YOUR KEY HERE (starts with gsk_)

# 4. OpenRouter - GATEWAY TO 400+ MODELS 🚀
#    Get key: https://openrouter.ai/settings/keys
#    Limits: 50 req/day on free models (20+ available)
#    Models: Llama 3.3 70B:free, Qwen3 235B:free, DeepSeek R1:free
# export OPENROUTER_API_KEY="sk-or-v1-your-key-here"  # <-- ADD YOUR KEY HERE (starts with sk-or-v1-)

# ============================================================================
# After adding keys, run:
#   source ~/BioPipelines/scripts/llm_api_keys.sh
#
# Then verify with:
#   cd ~/BioPipelines && python -c "
#   from workflow_composer.providers import get_registry
#   for p in get_registry().list_providers(configured_only=True):
#       print(f'{p.id}: ✅ configured')
#   "
# ============================================================================
