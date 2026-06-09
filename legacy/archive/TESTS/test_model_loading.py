"""Test improved model loading with quantization support."""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from swarm.llm.simple_llm import SimpleLLM
from swarm.core.config import MODEL_NAME, DEVICE

async def test_model_loading():
    """Test model loading with quantization."""
    print("=" * 70)
    print("TESTING MODEL LOADING WITH QUANTIZATION")
    print("=" * 70)
    print(f"\nModel: {MODEL_NAME}")
    print(f"Device: {DEVICE}")
    print(f"\nAttempting to load model with automatic quantization...\n")

    # Create LLM wrapper with auto-quantization
    llm = SimpleLLM(MODEL_NAME, DEVICE, enable_cache=False, use_quantization=None)

    # Load model
    await llm.load()

    # Test generation
    print("\n" + "=" * 70)
    print("TESTING GENERATION")
    print("=" * 70)

    prompt = "Write one sentence about artificial intelligence:"
    print(f"\nPrompt: {prompt}")
    print("Generating...\n")

    result = await llm.generate(prompt, max_tokens=50, temperature=0.7)

    if result:
        print(f"Generated: {result}")
        print("\nSUCCESS: Model loading and generation successful!")
    else:
        print("\nFAILED: Generation failed!")

    print("\n" + "=" * 70)

if __name__ == "__main__":
    asyncio.run(test_model_loading())
