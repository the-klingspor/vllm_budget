"""
Example usage of vllm_budget library.

This script demonstrates various ways to use the ThinkingBudgetLLM wrapper.
"""

from vllm_budget import ThinkingBudgetLLM
from vllm import SamplingParams


def example_basic_usage():
    """Basic usage with thinking budget."""
    print("=" * 60)
    print("Example 1: Basic Usage")
    print("=" * 60)
    
    # Initialize with thinking budget
    llm = ThinkingBudgetLLM(
        model="Qwen/Qwen3-1.7B",
        thinking_budget=200,
        tensor_parallel_size=1,
        enforce_eager=True,
    )
    
    # Create sampling parameters
    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=300,
        top_p=0.9
    )
    
    # Generate with thinking budget
    prompts = [
        "What is the capital of France?",
        "Explain quantum computing in simple terms."
    ]
    
    outputs = llm.generate(
        prompts=prompts,
        sampling_params=sampling_params
    )
    
    # Print results
    for prompt, output in zip(prompts, outputs):
        print(f"\nPrompt: {prompt}")
        print(f"Response: {output.outputs[0].text}")
        print("-" * 60)

    del llm  # Clean up


def example_custom_configuration():
    """Usage with custom configuration."""
    print("\n" + "=" * 60)
    print("Example 2: Custom Configuration")
    print("=" * 60)
    
    llm = ThinkingBudgetLLM(
        model="Qwen/Qwen3-1.7B",
        thinking_budget=200,
        early_stopping_text="\n\nBudget reached! Here's my answer:\n</think>\n\n",
        think_end_token="</think>",
        enforce_eager=True,
    )
    
    sampling_params = SamplingParams(temperature=0.8, max_tokens=300)
    
    outputs = llm.generate(
        prompts=["Solve the equation: 2x + 5 = 15"],
        sampling_params=sampling_params
    )
    
    print(f"Response: {outputs[0].outputs[0].text}")


def example_per_call_override():
    """Override thinking budget per call."""
    print("\n" + "=" * 60)
    print("Example 3: Per-Call Budget Override")
    print("=" * 60)
    
    llm = ThinkingBudgetLLM(
        model="Qwen/Qwen3-1.7B",
        thinking_budget=200,  # Default budget
        enforce_eager=True,
    )
    
    sampling_params = SamplingParams(temperature=0.7, max_tokens=400)
    
    # Use smaller budget for simple question
    simple_output = llm.generate(
        prompts=["What is 2+2?"],
        sampling_params=sampling_params,
        thinking_budget=20,  # Override with smaller budget
    )
    
    print(f"Simple question response: {simple_output[0].outputs[0].text}")
    
    # Use larger budget for complex question
    complex_output = llm.generate(
        prompts=["Explain the theory of relativity"],
        sampling_params=sampling_params,
        thinking_budget=300,  # Override with larger budget
    )
    
    print(f"\nComplex question response: {complex_output[0].outputs[0].text}")


def example_without_thinking_budget():
    """Standard generation without thinking budget."""
    print("\n" + "=" * 60)
    print("Example 4: Standard Generation (No Budget)")
    print("=" * 60)
    
    llm = ThinkingBudgetLLM(
        model="Qwen/Qwen3-1.7B",
        # No default thinking budget
        enforce_eager=True,
    )
    
    sampling_params = SamplingParams(temperature=0.7, max_tokens=300)
    
    # Generate without budget constraint
    outputs = llm.generate(
        prompts=["Write a haiku about programming"],
        sampling_params=sampling_params,
        thinking_budget=None,  # Explicitly no budget
    )
    
    print(f"Response: {outputs[0].outputs[0].text}")


def example_from_vllm_instance():
    """Create from existing vLLM instance."""
    print("\n" + "=" * 60)
    print("Example 5: From Existing vLLM Instance")
    print("=" * 60)
    
    from vllm import LLM
    
    # Create base vLLM instance with custom settings
    base_llm = LLM(
        model="Qwen/Qwen3-1.7B",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        enforce_eager=True,
    )
    
    # Wrap with thinking budget support
    llm = ThinkingBudgetLLM.from_vllm(
        base_llm,
        thinking_budget=200
    )
    
    sampling_params = SamplingParams(temperature=0.7, max_tokens=300)
    
    outputs = llm.generate(
        prompts=["What are the benefits of exercise?"],
        sampling_params=sampling_params
    )
    
    print(f"Response: {outputs[0].outputs[0].text}")


def example_batch_processing():
    """Process multiple prompts in batch."""
    print("\n" + "=" * 60)
    print("Example 6: Batch Processing")
    print("=" * 60)
    
    llm = ThinkingBudgetLLM(
        model="Qwen/Qwen3-1.7B",
        thinking_budget=200,
        enforce_eager=True,
    )
    
    sampling_params = SamplingParams(temperature=0.7, max_tokens=300)
    
    # Batch of prompts
    prompts = [
        "What is machine learning?",
        "How does photosynthesis work?",
        "Explain blockchain technology.",
        "What causes the seasons?"
    ]
    
    outputs = llm.generate(
        prompts=prompts,
        sampling_params=sampling_params
    )
    
    for i, (prompt, output) in enumerate(zip(prompts, outputs), 1):
        print(f"\n{i}. Prompt: {prompt}")
        print(f"   Response: {output.outputs[0].text[:100]}...")
        print("-" * 60)


def example_multiple_samples():
    """Generate multiple samples per prompt."""
    print("\n" + "=" * 60)
    print("Example 7: Multiple Samples (n>1)")
    print("=" * 60)
    
    llm = ThinkingBudgetLLM(
        model="Qwen/Qwen3-1.7B",
        thinking_budget=100,
        enforce_eager=True,
    )
    
    # Generate 3 different responses for same prompt
    sampling_params = SamplingParams(
        temperature=0.9,
        max_tokens=300,
        n=3,  # Generate 3 samples
    )
    
    outputs = llm.generate(
        prompts=["Tell me a creative story opening."],
        sampling_params=sampling_params
    )
    
    print("Generated 3 different story openings:\n")
    for i, sample in enumerate(outputs[0].outputs, 1):
        print(f"Sample {i}:")
        print(sample.text)
        print("-" * 60)


def example_tokenized_prompts():
    """Use pre-tokenized prompts (list of token IDs)."""
    print("\n" + "=" * 60)
    print("Example 8: Pre-Tokenized Prompts")
    print("=" * 60)
    
    llm = ThinkingBudgetLLM(
        model="Qwen/Qwen3-1.7B",
        thinking_budget=200,
        enforce_eager=True,
    )
    
    # Tokenize prompts manually
    text_prompts = [
        "What is the meaning of life?",
        "How do neural networks learn?"
    ]
    
    # Convert to token IDs
    tokenized_prompts = [
        {"prompt_token_ids": llm.tokenizer.encode(prompt)} for prompt in text_prompts
    ]
    
    print("Using pre-tokenized prompts:")
    for i, (text, tokens) in enumerate(zip(text_prompts, tokenized_prompts), 1):
        print(f"{i}. Text: {text}")
        print(f"   Token IDs: {tokens[:10]}... (showing first 10)")
        print()
    
    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=300
    )
    
    # Generate using token IDs instead of strings
    outputs = llm.generate(
        prompts=tokenized_prompts,
        sampling_params=sampling_params
    )
    
    # Print results
    for i, (text_prompt, output) in enumerate(zip(text_prompts, outputs), 1):
        print(f"{i}. Original prompt: {text_prompt}")
        print(f"   Response: {output.outputs[0].text}")
        print("-" * 60)


if __name__ == "__main__":
    # Run examples (comment out as needed)
    
    print("\n🚀 Starting vllm_budget examples...\n")
    
    try:
        example_basic_usage()
        example_custom_configuration()
        example_per_call_override()
        example_without_thinking_budget()
        example_from_vllm_instance()
        example_batch_processing()
        example_multiple_samples()
        example_tokenized_prompts()
        
        print("\n✅ All examples completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()