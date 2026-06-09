#!/usr/bin/env python3
"""Example usage of the new LLM architecture.

Demonstrates all 3 strategies:
1. Model Serving - Multiple backends (vLLM, SimpleLLM)
2. Agent Isolation - Provider pool with health checks
3. Request Batching - Automatic batching for throughput
"""

import asyncio
import logging
from swarm.llm.factory import LLMFactory
from swarm.llm.provider import GenerationRequest

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


async def example_simple_usage():
    """Simple usage with preset configuration."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Simple Usage with Preset")
    print("="*70 + "\n")

    # Use a preset configuration
    config = LLMFactory.create_config_preset("simple_fast")

    # Create system
    llm_system = await LLMFactory.create(config)
    await llm_system.start()

    # Generate text
    request = GenerationRequest(
        prompt="Explain quantum computing in one sentence.",
        max_tokens=50,
        temperature=0.8
    )

    print(f"Prompt: {request.prompt}")
    print("Generating...")

    response = await llm_system.generate(request)

    if response.success:
        print(f"\nResponse: {response.text}")
        print(f"Latency: {response.latency_ms:.1f}ms")
        print(f"Tokens: {response.tokens_generated}")
    else:
        print(f"Error: {response.error}")

    # Get system status
    status = llm_system.get_status()
    print(f"\nSystem Status:")
    print(f"  Healthy providers: {status['pool']['healthy_providers']}/{status['pool']['total_providers']}")

    # Cleanup
    await llm_system.stop()
    print("\nSystem stopped")


async def example_batch_processing():
    """Example of batch processing multiple requests."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Batch Processing")
    print("="*70 + "\n")

    # Configuration with batching enabled
    config = {
        "provider_type": "simple",
        "model_name": "Qwen/Qwen2.5-7B-Instruct",
        "device": "cuda",
        "enable_batching": True,
        "batch_size": 4,
        "batch_timeout_ms": 100.0,
        "use_quantization": True
    }

    llm_system = await LLMFactory.create(config)
    await llm_system.start()

    # Create multiple requests
    prompts = [
        "What is machine learning?",
        "Explain neural networks briefly.",
        "What is deep learning?",
        "Define artificial intelligence."
    ]

    requests = [
        GenerationRequest(prompt=p, max_tokens=30, temperature=0.7)
        for p in prompts
    ]

    print(f"Processing {len(requests)} requests...")
    print("(These will be automatically batched)\n")

    # Process all requests concurrently
    start_time = asyncio.get_event_loop().time()
    responses = await asyncio.gather(*[
        llm_system.generate(req) for req in requests
    ])
    elapsed = asyncio.get_event_loop().time() - start_time

    # Print results
    for i, (prompt, response) in enumerate(zip(prompts, responses)):
        print(f"[{i+1}] {prompt}")
        if response.success:
            print(f"    → {response.text[:100]}...")
            print(f"    Latency: {response.latency_ms:.1f}ms")
        else:
            print(f"    Error: {response.error}")
        print()

    print(f"Total time: {elapsed:.2f}s")
    print(f"Avg time per request: {elapsed/len(requests):.2f}s")

    # Show batcher metrics
    status = llm_system.get_status()
    if 'batcher' in status:
        batcher = status['batcher']
        print(f"\nBatcher Stats:")
        print(f"  Total batches: {batcher['total_batches']}")
        print(f"  Avg batch size: {batcher['avg_batch_size']:.1f}")

    await llm_system.stop()


async def example_provider_pool():
    """Example with provider pool for isolation."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Provider Pool with Isolation")
    print("="*70 + "\n")

    # Configuration with multiple providers
    config = {
        "provider_type": "simple",
        "model_name": "Qwen/Qwen2.5-7B-Instruct",
        "device": "cuda",
        "pool_size": 2,  # 2 isolated providers
        "health_check_interval": 30.0,
        "enable_batching": False,
        "use_quantization": True
    }

    llm_system = await LLMFactory.create(config)
    await llm_system.start()

    print("Created system with 2 isolated providers")
    print("If one fails, the other continues working\n")

    # Make several requests (will be load-balanced)
    for i in range(5):
        request = GenerationRequest(
            prompt=f"Count to {i+1}.",
            max_tokens=20
        )

        response = await llm_system.generate(request)

        if response.success:
            print(f"[{i+1}] Success: {response.text[:50]}...")
        else:
            print(f"[{i+1}] Failed: {response.error}")

    # Check pool status
    status = llm_system.get_status()
    print(f"\nPool Status:")
    for i, provider in enumerate(status['pool']['providers']):
        print(f"  Provider {i+1}:")
        print(f"    Type: {provider['type']}")
        print(f"    Healthy: {provider['healthy']}")
        print(f"    Load: {provider['current_load']}")
        print(f"    Circuit: {provider['circuit_breaker']['state']}")
        print(f"    Success rate: {provider['metrics']['success_rate']:.2%}")

    await llm_system.stop()


async def example_vllm_backend():
    """Example using vLLM backend (if available)."""
    print("\n" + "="*70)
    print("EXAMPLE 4: vLLM Backend (High Throughput)")
    print("="*70 + "\n")

    try:
        # Try to use vLLM
        config = LLMFactory.create_config_preset("vllm_optimal")

        llm_system = await LLMFactory.create(config)
        await llm_system.start()

        print("vLLM backend loaded successfully!")
        print("This provides 2-10x throughput vs SimpleLLM\n")

        # Make a request
        request = GenerationRequest(
            prompt="What are the benefits of using vLLM?",
            max_tokens=100,
            temperature=0.8
        )

        response = await llm_system.generate(request)

        if response.success:
            print(f"Response: {response.text}")
            print(f"Latency: {response.latency_ms:.1f}ms")
        else:
            print(f"Error: {response.error}")

        await llm_system.stop()

    except Exception as e:
        print(f"vLLM not available: {e}")
        print("Install with: pip install vllm")
        print("Falling back to SimpleLLM for this example")


async def example_monitoring():
    """Example of monitoring and metrics."""
    print("\n" + "="*70)
    print("EXAMPLE 5: Monitoring and Metrics")
    print("="*70 + "\n")

    config = LLMFactory.create_config_preset("simple_fast")
    llm_system = await LLMFactory.create(config)
    await llm_system.start()

    # Generate some requests
    for i in range(10):
        request = GenerationRequest(
            prompt=f"Count to {i+1}.",
            max_tokens=10
        )
        await llm_system.generate(request)

    # Get comprehensive status
    status = llm_system.get_status()

    print("System Metrics:\n")

    # Pool metrics
    pool = status['pool']
    print(f"Pool:")
    print(f"  Total providers: {pool['total_providers']}")
    print(f"  Healthy providers: {pool['healthy_providers']}")
    print()

    # Provider metrics
    for i, provider in enumerate(pool['providers']):
        print(f"Provider {i+1} ({provider['type']}):")
        metrics = provider['metrics']
        print(f"  Total requests: {metrics['total_requests']}")
        print(f"  Success rate: {metrics['success_rate']:.2%}")
        print(f"  Avg latency: {metrics['avg_latency_ms']:.1f}ms")
        print(f"  Avg tokens: {metrics['avg_tokens_per_request']:.1f}")
        print(f"  Health:")
        print(f"    Healthy: {metrics['health']['healthy']}")
        print(f"    Failures: {metrics['health']['failures']}")
        print(f"    Successes: {metrics['health']['successes']}")
        print()

    # Batcher metrics (if enabled)
    if status['batching_enabled'] and 'batcher' in status:
        batcher = status['batcher']
        print(f"Batcher:")
        print(f"  Total requests: {batcher['total_requests']}")
        print(f"  Total batches: {batcher['total_batches']}")
        print(f"  Avg batch size: {batcher['avg_batch_size']:.2f}")
        print(f"  Current queue: {batcher['current_queue_size']}/{batcher['max_queue_size']}")
        print()

    await llm_system.stop()


async def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("NEW LLM ARCHITECTURE - EXAMPLES")
    print("="*70)
    print("\nDemonstrating all 3 strategies:")
    print("  1. Model Serving - Multiple backends")
    print("  2. Agent Isolation - Provider pool")
    print("  3. Request Batching - Auto-batching")
    print()

    try:
        await example_simple_usage()
        await asyncio.sleep(1)

        await example_batch_processing()
        await asyncio.sleep(1)

        await example_provider_pool()
        await asyncio.sleep(1)

        await example_vllm_backend()
        await asyncio.sleep(1)

        await example_monitoring()

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*70)
    print("EXAMPLES COMPLETE")
    print("="*70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
