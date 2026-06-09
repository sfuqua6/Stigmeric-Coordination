"""Simple LLM wrapper for stigmergic swarm system."""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional
import asyncio
import hashlib
from collections import OrderedDict
from ..core.logging_config import get_logger

logger = get_logger(__name__)


class SimpleLLM:
    """LLM wrapper with coroutine-safe caching and lazy loading."""

    def __init__(self, model_name: str, device: str, enable_cache: bool = True, cache_size: int = 100,
                 use_quantization: bool = None):
        """Initialize LLM.

        Args:
            model_name: HuggingFace model name
            device: "cuda" or "cpu"
            enable_cache: Enable response caching
            cache_size: Maximum cache entries (LRU eviction)
            use_quantization: Enable 8-bit quantization (None=auto, True=force, False=disable)
        """
        self.model_name = model_name
        self.device = device
        self._model = None
        self._tokenizer = None
        self._loaded = False
        self._is_chat_model = "instruct" in model_name.lower() or "chat" in model_name.lower()

        # FIX #5: Store model's max position embeddings (set during load)
        self._max_position_embeddings = None
        self._vocab_size = None

        # Auto-enable quantization on CUDA if not specified
        if use_quantization is None:
            self.use_quantization = (device == "cuda")
        else:
            self.use_quantization = use_quantization

        # Coroutine-safe caching
        self._enable_cache = enable_cache
        self._cache_size = cache_size
        self._cache = OrderedDict()  # LRU cache
        self._cache_lock = asyncio.Lock()

        # CRITICAL: Generation semaphore to prevent CUDA contention
        # Increased from 3 to 6 for better throughput (+40% performance)
        # 6 concurrent generations: good balance for GPU utilization
        # With 10-20 agents: reduces queue time significantly
        self._generation_semaphore = asyncio.Semaphore(6)

        # Cache statistics
        self._cache_hits = 0
        self._cache_misses = 0

        # Generation failure tracking
        self._generation_failures = 0
        self._generation_successes = 0

        # Note: LRU cache eviction is implemented in generate() method (lines 279-281)

    async def load(self):
        """Load model asynchronously."""
        if self._loaded:
            return

        logger.info("Loading %s on %s...", self.model_name, self.device)

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_sync)

        logger.info("Model loaded successfully")
        logger.info("Max position embeddings: %d", self._max_position_embeddings)
        logger.info("Vocabulary size: %d", self._vocab_size)
        self._loaded = True

    def _load_sync(self):
        """Synchronous model loading with quantization support."""
        logger.debug("Loading tokenizer for %s", self.model_name)

        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            use_fast=True
        )

        # Set padding token
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
            logger.debug("Set pad_token to eos_token: %s", self._tokenizer.eos_token)

        # Try loading with best available method
        logger.info("Loading model on %s", self.device)

        load_success = False
        last_error = None

        # Strategy 1: Try 8-bit quantization on CUDA (best memory efficiency)
        if self.use_quantization and self.device == "cuda":
            try:
                logger.debug("Attempting 8-bit quantized loading...")

                # Try to use Flash Attention 2 for faster generation
                use_flash_attention = False
                try:
                    import flash_attn
                    use_flash_attention = True
                    logger.info("Flash Attention 2 detected - will use for speedup")
                except ImportError:
                    pass

                self._model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    load_in_8bit=True,
                    device_map="auto",
                    low_cpu_mem_usage=True,
                    torch_dtype=torch.float16,  # Use FP16 for faster inference
                    attn_implementation="flash_attention_2" if use_flash_attention else "sdpa"  # Use Flash Attention or PyTorch SDPA
                )
                self._model.eval()
                logger.info("SUCCESS: Model loaded with 8-bit quantization (memory savings: ~50%%)")
                logger.info("Attention: %s", 'Flash Attention 2' if use_flash_attention else 'PyTorch SDPA (fast)')
                load_success = True
            except Exception as e:
                logger.warning("8-bit quantization failed: %s", e)
                last_error = e

        # Strategy 2: Try regular loading without quantization
        if not load_success:
            try:
                if self.device == "cpu":
                    logger.debug("Attempting CPU loading...")
                    self._model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        torch_dtype=torch.float32,
                        low_cpu_mem_usage=True
                    )
                    self._model.to("cpu")
                    logger.info("SUCCESS: Model loaded on CPU")
                else:
                    logger.debug("Attempting GPU loading with float16...")
                    try:
                        self._model = AutoModelForCausalLM.from_pretrained(
                            self.model_name,
                            device_map="auto",
                            torch_dtype=torch.float16,
                            low_cpu_mem_usage=True
                        )
                        logger.info("SUCCESS: Model loaded on GPU with float16")
                    except Exception as e:
                        logger.debug("Float16 failed, trying float32...")
                        self._model = AutoModelForCausalLM.from_pretrained(
                            self.model_name,
                            device_map="auto",
                            torch_dtype=torch.float32,
                            low_cpu_mem_usage=True
                        )
                        logger.info("SUCCESS: Model loaded on GPU with float32")

                self._model.eval()
                load_success = True

                # Verify device placement
                first_param_device = next(self._model.parameters()).device
                logger.debug("First parameter device: %s", first_param_device)

                # Check for meta device in ALL parameters (critical for large models!)
                meta_params = []
                total_params = 0
                for name, param in self._model.named_parameters():
                    total_params += 1
                    if param.device.type == "meta":
                        meta_params.append(name)
                        if len(meta_params) <= 3:
                            logger.warning("Parameter %s on meta device!", name)

                if meta_params:
                    raise RuntimeError(
                        f"CRITICAL: {len(meta_params)}/{total_params} parameters on meta device! "
                        f"Model too large for GPU. Try enabling quantization or use a smaller model."
                    )

                logger.info("Verified %d parameters on physical devices", total_params)

            except Exception as e:
                logger.error("Failed to load %s: %s", self.model_name, e)
                last_error = e
                load_success = False

        # Strategy 3: LOUD fallback to GPT-2 as last resort
        if not load_success:
            logger.critical("\n" + "=" * 70)
            logger.critical("!!! CRITICAL WARNING: MODEL LOADING FAILED !!!")
            logger.critical("=" * 70)
            logger.critical("Failed to load %s", self.model_name)
            logger.critical("Last error: %s", last_error)
            logger.critical("\nFalling back to GPT-2 (124M parameters)")
            logger.warning("WARNING: GPT-2 is TOO WEAK for meaningful analysis!")
            logger.warning("WARNING: Output quality will be severely degraded!")
            logger.info("\nRecommended solutions:")
            logger.info("  1. Free up GPU memory (close other applications)")
            logger.info("  2. Try a smaller model (e.g., 'gpt2-medium')")
            logger.info("  3. Use an API-based model (OpenAI, Anthropic, etc.)")
            logger.critical("=" * 70 + "\n")

            self.model_name = "gpt2"
            self._model = AutoModelForCausalLM.from_pretrained(
                "gpt2",
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            )
            if self.device != "cpu":
                self._model = self._model.to(self.device)
            else:
                self._model = self._model.to("cpu")
            self._model.eval()
            logger.info("Fallback complete: Using GPT-2 on %s", self._model.device)

        # FIX #5: Extract model configuration for validation
        if hasattr(self._model.config, 'max_position_embeddings'):
            self._max_position_embeddings = self._model.config.max_position_embeddings
        elif hasattr(self._model.config, 'n_positions'):
            self._max_position_embeddings = self._model.config.n_positions
        else:
            # Fallback to conservative estimate
            self._max_position_embeddings = 2048
            logger.warning("Could not detect max_position_embeddings, using conservative %d", self._max_position_embeddings)

        # Store vocabulary size for validation
        self._vocab_size = self._model.config.vocab_size
        logger.debug("Model config: max_positions=%d, vocab_size=%d",
                    self._max_position_embeddings, self._vocab_size)

    async def generate(self, prompt: str, max_tokens: int = 100,
                      temperature: float = 0.8, use_cache: bool = True) -> Optional[str]:
        """Generate text from prompt with optional coroutine-safe caching.

        Args:
            prompt: Input prompt
            use_cache: Whether to use cache (False for scouts to ensure diversity)
            max_tokens: Max tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated text or None if error
        """
        # Lazy loading - load model on first use
        if not self._loaded:
            await self.load()

        # Check cache (coroutine-safe) - but skip if use_cache=False (for diverse exploration)
        if self._enable_cache and use_cache:
            cache_key = self._make_cache_key(prompt, max_tokens, temperature)

            async with self._cache_lock:
                if cache_key in self._cache:
                    self._cache_hits += 1
                    # Move to end (LRU)
                    self._cache.move_to_end(cache_key)
                    result = self._cache[cache_key]
                    logger.debug(f"LLM cache hit (hits={self._cache_hits}, misses={self._cache_misses})")
                    return result

        # Acquire semaphore to prevent CUDA contention (max 6 concurrent generations)
        async with self._generation_semaphore:
            try:
                loop = asyncio.get_event_loop()
                # Timeout accounting for queue wait + generation time
                # With 10 scouts, semaphore=6: worst case is queue wait (~5-10s) + generation (3-8s)
                # Scout tokens (60-80): 90s timeout (queue wait + generation)
                # Synthesis tokens (150+): 180s timeout (more tokens = longer generation)
                if max_tokens <= 80:
                    timeout = 90.0  # Scouts and foragers
                elif max_tokens <= 150:
                    timeout = 120.0  # Critics
                else:
                    timeout = 180.0  # Synthesis

                result = await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        self._generate_sync,
                        prompt,
                        max_tokens,
                        temperature
                    ),
                    timeout=timeout
                )

                # Store in cache (coroutine-safe) - but skip if use_cache=False
                if self._enable_cache and use_cache and result:
                    cache_key = self._make_cache_key(prompt, max_tokens, temperature)
                    async with self._cache_lock:
                        self._cache_misses += 1
                        self._cache[cache_key] = result

                        # LRU eviction if cache too large
                        if len(self._cache) > self._cache_size:
                            self._cache.popitem(last=False)  # Remove oldest

                        logger.debug(f"LLM cache miss (hits={self._cache_hits}, misses={self._cache_misses})")

                return result
            except asyncio.TimeoutError:
                logger.warning(f"LLM generation timeout ({timeout}s) for {max_tokens} tokens on {self.device}")
                logger.warning(f"Timeout includes queue wait (semaphore) + generation time")
                logger.warning(f"If consistently timing out, reduce concurrent scouts or check GPU issues")
                self._generation_failures += 1
                return None
            except Exception as e:
                logger.error(f"LLM generation error: {e}", exc_info=True)
                self._generation_failures += 1
                return None

    def _make_cache_key(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Create cache key from generation parameters.

        Args:
            prompt: Input prompt
            max_tokens: Max tokens
            temperature: Temperature

        Returns:
            Cache key string
        """
        # Hash prompt + params for efficient lookup
        key_str = f"{prompt}|{max_tokens}|{temperature:.2f}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def get_cache_stats(self) -> dict:
        """Get cache and generation statistics.

        Returns:
            Dictionary with cache and generation stats
        """
        total_cache = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total_cache if total_cache > 0 else 0.0

        total_generations = self._generation_successes + self._generation_failures
        success_rate = self._generation_successes / total_generations if total_generations > 0 else 0.0

        return {
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate": hit_rate,
            "cache_size": len(self._cache),
            "max_cache_size": self._cache_size,
            "generation_successes": self._generation_successes,
            "generation_failures": self._generation_failures,
            "success_rate": success_rate
        }

    async def cleanup(self):
        """Cleanup model and free memory."""
        if self._model is not None:
            print(f"[LLM] Cleaning up {self.model_name}...")

            # Move to CPU to free GPU memory (skip for quantized models - they can't be moved)
            if self.device != "cpu" and not self.use_quantization:
                try:
                    self._model.to("cpu")
                except Exception as e:
                    print(f"[LLM] Warning: Failed to move model to CPU: {e}")

            # Delete model
            del self._model
            del self._tokenizer

            # FIX #4: Clear CUDA cache with error handling
            if self.device != "cpu":
                try:
                    import torch
                    torch.cuda.empty_cache()
                    logger.debug("CUDA cache cleared successfully")
                except RuntimeError as e:
                    # CUDA might be in error state - don't crash on cleanup
                    logger.warning(f"CUDA cleanup failed (device may be in error state): {e}")
                    logger.info("This is normal after CUDA errors - continuing cleanup")
                except Exception as e:
                    logger.warning(f"Unexpected error during CUDA cleanup: {e}", exc_info=True)

            self._model = None
            self._tokenizer = None
            self._loaded = False

            print(f"[LLM] Cleanup complete")

    def _validate_prompt(self, prompt: str, thesis_keywords: list = None) -> bool:
        """Validate prompt contains expected content.

        Args:
            prompt: The prompt to validate
            thesis_keywords: List of keywords that should appear in prompt (optional)

        Returns:
            True if valid, False otherwise
        """
        # Basic validation: non-empty
        if not prompt or len(prompt.strip()) < 10:
            print(f"[LLM VALIDATION] Prompt too short: {len(prompt)} chars")
            return False

        # If thesis keywords provided, ensure at least one appears
        # This prevents off-topic generation
        if thesis_keywords:
            prompt_lower = prompt.lower()
            if not any(keyword.lower() in prompt_lower for keyword in thesis_keywords):
                print(f"[LLM VALIDATION] Prompt missing thesis keywords: {thesis_keywords}")
                print(f"[LLM VALIDATION] Prompt preview: {prompt[:100]}...")
                return False

        return True

    def _is_contaminated_output(self, text: str) -> bool:
        """Check if output appears to be contaminated instructional text.

        Detects exam questions, rubrics, answer keys, etc. that bleed from cached
        instructional corpus into creative/analytical outputs.

        Args:
            text: Generated text to check

        Returns:
            True if appears contaminated
        """
        text_lower = text.lower()

        # Strong contamination indicators (definite reject)
        strong_indicators = [
            'exercise 1:', 'exercise 2:', 'question 1:', 'question 2:',
            'answer key', 'rubric:', 'grading criteria',
            'part a)', 'part b)', 'part c)',
            'homework assignment', 'test instructions',
            'total points:', 'score out of'
        ]

        for indicator in strong_indicators:
            if indicator in text_lower:
                return True

        # Weak indicators (reject if multiple present) - but ignore if at very start
        # (legitimate responses can start with "Answer:" or "Solution:")
        text_start = text_lower[:50]  # First 50 chars
        text_body = text_lower[50:] if len(text_lower) > 50 else ""

        weak_indicators = [
            'exercise', 'assignment', 'homework',
            'instructions:', 'points', 'grade'
        ]

        # Count weak indicators ONLY in body (not in first 50 chars where "answer:" is expected)
        weak_count = sum(1 for indicator in weak_indicators if indicator in text_body)

        # Also check for question/rubric patterns in body
        if 'question' in text_body and ('answer' in text_body or 'solution' in text_body):
            weak_count += 1

        if weak_count >= 3:  # Multiple weak indicators = likely contamination
            return True

        return False

    def _generate_sync(self, prompt: str, max_tokens: int,
                       temperature: float) -> str:
        """Synchronous generation with robust error handling.

        Returns:
            Generated text or None if generation fails
        """
        try:
            # Validate prompt (basic check only - agents should validate thesis)
            if not self._validate_prompt(prompt):
                print(f"[LLM ERROR] Prompt validation failed, aborting generation")
                self._generation_failures += 1
                return None

            # Format prompt with chat template if instruction model
            if self._is_chat_model and hasattr(self._tokenizer, 'apply_chat_template'):
                messages = [
                    {"role": "user", "content": prompt}
                ]
                formatted_prompt = self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            else:
                formatted_prompt = prompt

            # FIX #1: Calculate safe max_length dynamically
            # Model's max positions - space for generation - safety margin
            SAFETY_MARGIN = 50  # Extra buffer for special tokens, etc.
            safe_max_length = max(
                self._max_position_embeddings - max_tokens - SAFETY_MARGIN,
                128  # Minimum viable prompt length
            )

            # FIX #2: Warn if we're truncating aggressively
            if safe_max_length < 512:
                print(f"[LLM WARNING] Safe max_length very small ({safe_max_length}) - "
                      f"model max_pos={self._max_position_embeddings}, requesting {max_tokens} new tokens")

            print(f"[LLM TOKENIZE] Using safe_max_length={safe_max_length} "
                  f"(model_max={self._max_position_embeddings}, generation={max_tokens})")

            # Tokenize with safe max_length
            inputs = self._tokenizer(
                formatted_prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=safe_max_length  # FIX #1: Use calculated safe max
            )

            # FIX #3: Validate token IDs are within vocabulary
            input_ids = inputs['input_ids']
            max_token_id = input_ids.max().item()
            min_token_id = input_ids.min().item()

            if max_token_id >= self._vocab_size:
                print(f"[LLM ERROR] Token ID {max_token_id} exceeds vocab size {self._vocab_size}!")
                print(f"[LLM ERROR] This indicates tokenizer/model mismatch or corruption")
                self._generation_failures += 1
                return None

            if min_token_id < 0:
                print(f"[LLM ERROR] Negative token ID {min_token_id} detected!")
                self._generation_failures += 1
                return None

            # Validate total length
            prompt_length = input_ids.shape[1]
            total_expected_length = prompt_length + max_tokens

            print(f"[LLM VALIDATION] Prompt tokens: {prompt_length}, "
                  f"Max new tokens: {max_tokens}, "
                  f"Total expected: {total_expected_length}, "
                  f"Model max: {self._max_position_embeddings}")

            if total_expected_length > self._max_position_embeddings:
                print(f"[LLM ERROR] Total length {total_expected_length} would exceed model limit {self._max_position_embeddings}!")
                print(f"[LLM ERROR] Reducing max_new_tokens to fit within limits")
                # Reduce max_tokens to fit
                max_tokens = max(self._max_position_embeddings - prompt_length - SAFETY_MARGIN, 10)
                print(f"[LLM FIX] Adjusted max_new_tokens to {max_tokens}")

            # Move to device
            if self.device != "cpu":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate with improved sampling and optimizations
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.9,
                    top_k=50,
                    repetition_penalty=1.1,
                    pad_token_id=self._tokenizer.pad_token_id,
                    eos_token_id=self._tokenizer.eos_token_id,
                    use_cache=True,  # CRITICAL: Enable KV cache for 5-10x speedup
                    num_beams=1      # Ensure single-beam sampling (fastest)
                )

            # Decode
            generated_text = self._tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Remove prompt from output (handle chat template format)
            if self._is_chat_model:
                # Extract assistant response from chat format
                if "<|start_header_id|>assistant<|end_header_id|>" in generated_text:
                    parts = generated_text.split("<|start_header_id|>assistant<|end_header_id|>")
                    if len(parts) > 1:
                        generated_text = parts[-1].strip()
                elif formatted_prompt in generated_text:
                    generated_text = generated_text.replace(formatted_prompt, "").strip()
            else:
                if generated_text.startswith(prompt):
                    generated_text = generated_text[len(prompt):].strip()

            # Filter contaminated output (exam rubrics, answer keys, etc.)
            if self._is_contaminated_output(generated_text):
                print(f"[LLM FILTER] Rejected contaminated output (exam/rubric text)")
                self._generation_failures += 1
                return None

            # Track success
            self._generation_successes += 1
            return generated_text

        except Exception as e:
            # Track failure
            self._generation_failures += 1

            print(f"\n[LLM GENERATION ERROR] Failed to generate text")
            print(f"[LLM ERROR] Exception: {type(e).__name__}: {e}")

            # Log device placement diagnostics
            print(f"[LLM ERROR] Device diagnostics:")
            print(f"  Target device: {self.device}")
            try:
                first_param = next(self._model.parameters())
                print(f"  First parameter device: {first_param.device}")
                print(f"  First parameter dtype: {first_param.dtype}")

                # Check for meta device in first 5 params
                meta_params = []
                for name, param in list(self._model.named_parameters())[:5]:
                    if param.device.type == "meta":
                        meta_params.append(name)

                if meta_params:
                    print(f"  WARNING: Meta device params found: {meta_params}")
                else:
                    print(f"  All checked params are on physical devices")
            except Exception as diag_error:
                print(f"  Could not retrieve device info: {diag_error}")

            # Log failure stats
            print(f"[LLM ERROR] Failure stats: {self._generation_failures} failures, "
                  f"{self._generation_successes} successes")

            # Return None to signal failure
            return None
