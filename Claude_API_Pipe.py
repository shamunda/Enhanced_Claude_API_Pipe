"""
title: Enhanced Claude API Pipe
author: Dennis (Shamunda) Ross
author_url: https://github.com/shamunda
version: 1.0.0
required_open_webui_version: 0.5.00
license: MIT
"""

import os
import time
import json
import random
import asyncio
import logging
from datetime import datetime
import httpx
from typing import List, Union, Generator, Iterator, Dict, Any, Optional
from pydantic import BaseModel, Field, validator
from open_webui.utils.misc import pop_system_message


class EnhancedPipe:
    """
    Advanced implementation of an Anthropic Claude API connector for Open WebUI.
    Provides robust error handling, advanced configuration, and performance optimizations.
    """

    class Valves(BaseModel):
        """Valve configuration with detailed descriptions"""

        ANTHROPIC_API_KEY: str = Field(default="", description="Anthropic API key")
        STREAMING_MODE: bool = Field(
            default=True, description="Enable streaming mode for faster responses."
        )
        VERBOSE_LOGGING: bool = Field(
            default=False, description="Enable verbose logging for debugging."
        )
        MAX_TOKENS: int = Field(
            default=4096,
            description="Maximum number of tokens to generate. Reduce to limit response length and rate limit issues.",
            ge=1,
            le=100000,
        )
        REQUESTS_PER_MINUTE: int = Field(
            default=150,
            description="Maximum requests per minute. Start conservatively.",
            ge=1,
        )
        MAX_RETRIES: int = Field(
            default=5,
            description="Maximum number of retries for API requests",
            ge=0,
            le=10,
        )
        INITIAL_RETRY_DELAY: float = Field(
            default=1.0,
            description="Initial delay in seconds before retrying a failed request",
            ge=0.1,
        )
        MAX_RETRY_DELAY: float = Field(
            default=30.0, description="Maximum delay in seconds between retries", ge=1.0
        )
        RETRY_JITTER: float = Field(
            default=0.1,
            description="Random jitter factor to add to retry delays",
            ge=0,
            le=0.5,
        )
        IMAGE_SIZE_LIMIT: int = Field(
            default=5 * 1024 * 1024,  # 5MB
            description="Maximum size for individual images in bytes",
            ge=1024,
        )
        TOTAL_IMAGE_LIMIT: int = Field(
            default=100 * 1024 * 1024,  # 100MB
            description="Maximum total size for all images in bytes",
            ge=1024,
        )
        RESPONSE_TIMEOUT: float = Field(
            default=120.0,
            description="Maximum seconds to wait for API response",
            ge=1.0,
        )
        CONNECTION_TIMEOUT: float = Field(
            default=10.0, description="Maximum seconds to wait for connection", ge=1.0
        )

        # Remove all validators entirely - validation happens in __init__

    def __init__(self):
        """Initialize the pipe with configuration and set up logging"""
        self.type = "manifold"
        self.id = "anthropic-enhanced"
        self.name = "anthropic-enhanced/"

        # Initialize logger
        self.logger = self._setup_logger()

        # Load and validate valves from environment
        self.valves = self._load_valves()

        # Rate limiting tracking
        self.request_timestamps = []
        self.last_log_time = 0

        # HTTP client for better connection pooling
        self.client = httpx.Client(
            timeout=httpx.Timeout(
                connect=self.valves.CONNECTION_TIMEOUT,
                read=self.valves.RESPONSE_TIMEOUT,
                pool=5.0,
                write=5.0,
            )
        )

        # Warn if no API key is set - moved here from validator
        if not self.valves.ANTHROPIC_API_KEY:
            self.logger.warning(
                "ANTHROPIC_API_KEY is not set. Please set the environment variable or Valve."
            )

        self.logger.info(f"Enhanced Claude API Pipe initialized (version 1.0.0)")

    def _setup_logger(self) -> logging.Logger:
        """Configure a custom logger with formatting"""
        logger = logging.getLogger("EnhancedClaudePipe")

        # Set log level from environment or default to INFO
        log_level = os.getenv("LOG_LEVEL", "INFO").upper()
        logger.setLevel(getattr(logging, log_level, logging.INFO))

        # Create console handler with formatting
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s [%(levelname)s] EnhancedClaudePipe: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def _load_valves(self) -> Valves:
        """Load valves from environment variables with fallbacks"""
        valves_dict = {
            "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY", ""),
            "STREAMING_MODE": os.getenv("STREAMING_MODE", "true").lower() == "true",
            "VERBOSE_LOGGING": os.getenv("VERBOSE_LOGGING", "false").lower() == "true",
            "MAX_TOKENS": int(os.getenv("MAX_TOKENS", "4096")),
            "REQUESTS_PER_MINUTE": int(os.getenv("REQUESTS_PER_MINUTE", "150")),
            "MAX_RETRIES": int(os.getenv("MAX_RETRIES", "5")),
            "INITIAL_RETRY_DELAY": float(os.getenv("INITIAL_RETRY_DELAY", "1.0")),
            "MAX_RETRY_DELAY": float(os.getenv("MAX_RETRY_DELAY", "30.0")),
            "RETRY_JITTER": float(os.getenv("RETRY_JITTER", "0.1")),
            "IMAGE_SIZE_LIMIT": int(
                os.getenv("IMAGE_SIZE_LIMIT", str(5 * 1024 * 1024))
            ),
            "TOTAL_IMAGE_LIMIT": int(
                os.getenv("TOTAL_IMAGE_LIMIT", str(100 * 1024 * 1024))
            ),
            "RESPONSE_TIMEOUT": float(os.getenv("RESPONSE_TIMEOUT", "120.0")),
            "CONNECTION_TIMEOUT": float(os.getenv("CONNECTION_TIMEOUT", "10.0")),
        }

        return self.Valves(**valves_dict)

    def pipes(self) -> List[dict]:
        """Return available models for the UI selector"""
        return [
            {"id": "claude-3-5-haiku-latest", "name": "Claude 3.5 Haiku"},
            {"id": "claude-3-5-sonnet-latest", "name": "Claude 3.5 Sonnet"},
            {
                "id": "claude-3-7-sonnet-20250219",
                "name": "Claude 3.7 Sonnet (Feb 2025)",
            },
            {"id": "claude-3-7-sonnet-latest", "name": "Claude 3.7 Sonnet (Latest)"},
            {"id": "claude-3-opus-latest", "name": "Claude 3 Opus"},
        ]

    def debug(self, message: str):
        """Log debug messages only when verbose logging is enabled"""
        if self.valves.VERBOSE_LOGGING:
            self.logger.debug(message)

    def _manage_rate_limit(self):
        """Advanced rate limiting with dynamic adjustments"""
        current_time = time.time()

        # Remove timestamps older than 60 seconds
        self.request_timestamps = [
            t for t in self.request_timestamps if current_time - t < 60
        ]

        # Check if we've hit the rate limit
        if len(self.request_timestamps) >= self.valves.REQUESTS_PER_MINUTE:
            # Calculate wait time - add a small buffer to be safe
            oldest_timestamp = min(self.request_timestamps)
            wait_time = 60 - (current_time - oldest_timestamp) + 0.5

            self.logger.warning(
                f"Rate limit reached. Waiting {wait_time:.2f}s before next request"
            )
            time.sleep(max(0, wait_time))

            # After waiting, refresh timestamps
            self.request_timestamps = [
                t for t in self.request_timestamps if time.time() - t < 60
            ]

        # Add current request to timestamps
        self.request_timestamps.append(time.time())

        # Log rate limit usage periodically (at most once every 15 seconds)
        if current_time - self.last_log_time > 15:
            self.debug(
                f"Rate limit usage: {len(self.request_timestamps)}/{self.valves.REQUESTS_PER_MINUTE} requests in last minute"
            )
            self.last_log_time = current_time

    def _calculate_backoff(self, attempt: int) -> float:
        """Calculate exponential backoff with jitter for retries"""
        # Base exponential backoff
        delay = min(
            self.valves.MAX_RETRY_DELAY, self.valves.INITIAL_RETRY_DELAY * (2**attempt)
        )

        # Add jitter to prevent thundering herd
        max_jitter = delay * self.valves.RETRY_JITTER
        jitter = random.uniform(-max_jitter, max_jitter)

        return max(0.1, delay + jitter)

    def _process_image(self, image_data: dict) -> dict:
        """Process and validate image data for API submission"""
        self.debug(
            f"Processing image: {image_data.get('image_url', {}).get('url', '')[:30]}..."
        )

        if image_data["image_url"]["url"].startswith("data:image"):
            # Handle base64 encoded images
            try:
                mime_type, base64_data = image_data["image_url"]["url"].split(",", 1)
                media_type = mime_type.split(":")[1].split(";")[0]

                # Calculate image size from base64
                image_size = (
                    len(base64_data) * 3 / 4
                )  # Approximate base64 to bytes conversion

                if image_size > self.valves.IMAGE_SIZE_LIMIT:
                    raise ValueError(
                        f"Image exceeds size limit: {image_size / 1024 / 1024:.2f}MB "
                        f"(max: {self.valves.IMAGE_SIZE_LIMIT / 1024 / 1024}MB)"
                    )

                return {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": base64_data,
                    },
                }
            except Exception as e:
                self.logger.error(f"Failed to process base64 image: {str(e)}")
                raise ValueError(f"Failed to process image: {str(e)}")
        else:
            # Handle URL images
            url = image_data["image_url"]["url"]
            try:
                response = self.client.head(url, follow_redirects=True)
                response.raise_for_status()

                # Check content length if available
                content_length = int(response.headers.get("content-length", 0))
                if content_length > 0 and content_length > self.valves.IMAGE_SIZE_LIMIT:
                    raise ValueError(
                        f"Image at URL exceeds size limit: {content_length / 1024 / 1024:.2f}MB "
                        f"(max: {self.valves.IMAGE_SIZE_LIMIT / 1024 / 1024}MB)"
                    )

                return {
                    "type": "image",
                    "source": {"type": "url", "url": url},
                }
            except httpx.HTTPStatusError as e:
                self.logger.error(
                    f"Failed to access image URL ({url}): HTTP {e.response.status_code}"
                )
                raise ValueError(
                    f"Failed to access image URL: HTTP {e.response.status_code}"
                )
            except Exception as e:
                self.logger.error(f"Failed to process image URL ({url}): {str(e)}")
                raise ValueError(f"Failed to process image URL: {str(e)}")

    def _prepare_messages(self, messages: List[dict]) -> tuple:
        """Process messages and extract system message"""
        system_message, user_messages = pop_system_message(messages)
        processed_messages = []
        total_image_size = 0

        # Process each message
        for message in user_messages:
            processed_content = []

            # Handle content based on type
            if isinstance(message.get("content"), list):
                for item in message["content"]:
                    if item["type"] == "text":
                        processed_content.append({"type": "text", "text": item["text"]})
                    elif item["type"] == "image_url":
                        img_data = self._process_image(item)
                        processed_content.append(img_data)

                        # Track total image size
                        if img_data["source"]["type"] == "base64":
                            image_size = len(img_data["source"]["data"]) * 3 / 4
                            total_image_size += image_size

                            if total_image_size > self.valves.TOTAL_IMAGE_LIMIT:
                                raise ValueError(
                                    f"Total image size ({total_image_size / 1024 / 1024:.2f}MB) "
                                    f"exceeds limit ({self.valves.TOTAL_IMAGE_LIMIT / 1024 / 1024}MB)"
                                )
            else:
                # Handle simple text content
                processed_content = [
                    {"type": "text", "text": message.get("content", "")}
                ]

            processed_messages.append(
                {"role": message["role"], "content": processed_content}
            )

        return system_message, processed_messages

    def pipe(self, body: dict) -> Union[str, Generator, Iterator]:
        """Main entry point for processing requests from Open WebUI"""
        request_id = f"req_{int(time.time() * 1000)}"
        start_time = time.time()

        self.logger.info(
            f"[{request_id}] Processing request for model: {body.get('model', 'unknown')}"
        )
        self.debug(
            f"[{request_id}] Request body: {json.dumps(body, default=str)[:500]}..."
        )

        # Determine if streaming based on valves and request
        stream_mode = self.valves.STREAMING_MODE and body.get("stream", False)

        try:
            # Process messages
            system_message, processed_messages = self._prepare_messages(
                body["messages"]
            )

            # Get model ID, removing prefix if needed
            model_id = body["model"]
            if "." in model_id:
                model_id = model_id[model_id.find(".") + 1 :]

            # Build API request payload
            payload = {
                "model": model_id,
                "messages": processed_messages,
                "max_tokens": min(
                    int(body.get("max_tokens", self.valves.MAX_TOKENS)),
                    self.valves.MAX_TOKENS,
                ),
                "temperature": body.get("temperature", 0.7),
                "top_k": body.get("top_k", 40),
                "top_p": body.get("top_p", 0.95),
                "stop_sequences": body.get("stop", []),
                "stream": stream_mode,
            }

            # Add system message if provided
            if system_message:
                payload["system"] = str(system_message)

            # Set up request headers
            headers = {
                "x-api-key": self.valves.ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
                "user-agent": "OpenWebUI-EnhancedClaudePipe/1.0.0",
            }

            # Add client info if available
            if "client" in body:
                headers["client-info"] = body["client"]

            # API endpoint
            api_url = "https://api.anthropic.com/v1/messages"

            # Execute request with retries
            for attempt in range(self.valves.MAX_RETRIES + 1):
                try:
                    # Apply rate limiting
                    self._manage_rate_limit()

                    # Make API call based on stream mode
                    if stream_mode:
                        return self._handle_streaming(
                            request_id, api_url, headers, payload
                        )
                    else:
                        return self._handle_non_streaming(
                            request_id, api_url, headers, payload
                        )

                except httpx.HTTPStatusError as e:
                    status_code = e.response.status_code
                    error_text = e.response.text

                    # Special handling for specific status codes
                    if status_code == 429:
                        self.logger.warning(
                            f"[{request_id}] Rate limit exceeded (429): {error_text}"
                        )
                    elif status_code == 401:
                        self.logger.error(
                            f"[{request_id}] Authentication failed (401): Check API key"
                        )
                        return (
                            "Error: Authentication failed. Please check your API key."
                        )
                    elif status_code == 400:
                        self.logger.error(
                            f"[{request_id}] Bad request (400): {error_text}"
                        )
                        # Don't retry on bad requests
                        return f"Error: Bad request - {error_text}"
                    else:
                        self.logger.error(
                            f"[{request_id}] HTTP error {status_code}: {error_text}"
                        )

                    # Check if we should retry
                    if attempt < self.valves.MAX_RETRIES:
                        delay = self._calculate_backoff(attempt)
                        self.logger.warning(
                            f"[{request_id}] Retrying in {delay:.2f}s "
                            f"(attempt {attempt+1}/{self.valves.MAX_RETRIES})"
                        )
                        time.sleep(delay)
                    else:
                        return f"Error: Request failed after {self.valves.MAX_RETRIES} attempts. Last error: {error_text}"

                except Exception as e:
                    self.logger.error(f"[{request_id}] Request error: {str(e)}")

                    # Check if we should retry
                    if attempt < self.valves.MAX_RETRIES:
                        delay = self._calculate_backoff(attempt)
                        self.logger.warning(
                            f"[{request_id}] Retrying in {delay:.2f}s "
                            f"(attempt {attempt+1}/{self.valves.MAX_RETRIES})"
                        )
                        time.sleep(delay)
                    else:
                        return f"Error: Request failed after {self.valves.MAX_RETRIES} attempts. {str(e)}"

            # This shouldn't be reached but just in case
            return "Error: Request failed after exhausting all retry attempts."

        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(
                f"[{request_id}] Error processing request ({duration:.2f}s): {str(e)}"
            )
            return f"Error: {str(e)}"

    def _handle_streaming(
        self, request_id: str, url: str, headers: dict, payload: dict
    ) -> Generator[str, None, None]:
        """Handle streaming API responses"""
        self.debug(f"[{request_id}] Starting streaming request")

        try:
            with self.client.stream(
                "POST", url, headers=headers, json=payload, timeout=None
            ) as response:
                response.raise_for_status()

                # Track when we last received data to detect stalls
                last_data_time = time.time()

                # Process stream data
                for line in response.iter_lines():
                    if not line:
                        continue

                    line = line.strip()
                    if not line.startswith(b"data: "):
                        continue

                    try:
                        # Update data timestamp
                        last_data_time = time.time()

                        # Extract and parse JSON
                        data_str = line[6:].decode("utf-8")
                        data = json.loads(data_str)

                        # Process different event types
                        if data["type"] == "content_block_start":
                            yield data["content_block"]["text"]
                        elif data["type"] == "content_block_delta":
                            yield data["delta"]["text"]
                        elif data["type"] == "message_stop":
                            self.debug(
                                f"[{request_id}] Streaming complete (message_stop)"
                            )
                            break
                        elif data["type"] == "message":
                            for content in data.get("content", []):
                                if content["type"] == "text":
                                    yield content["text"]

                        # Tiny delay to avoid client flooding
                        time.sleep(0.005)

                    except json.JSONDecodeError:
                        self.logger.warning(
                            f"[{request_id}] Failed to parse JSON: {line}"
                        )
                    except KeyError as e:
                        self.logger.warning(
                            f"[{request_id}] Unexpected data structure: {e}"
                        )
                    except Exception as e:
                        self.logger.error(
                            f"[{request_id}] Error processing stream data: {e}"
                        )

                    # Check for stream stall (no data for 30 seconds)
                    if time.time() - last_data_time > 30:
                        self.logger.warning(
                            f"[{request_id}] Stream stalled (no data for 30s)"
                        )
                        yield "\n[Error: Connection stalled. Please try again.]"
                        break

        except httpx.HTTPStatusError as e:
            self.logger.error(
                f"[{request_id}] HTTP error during streaming: {e.response.status_code}"
            )
            yield f"\n[Error: HTTP {e.response.status_code}: {e.response.text}]"
            raise e

        except Exception as e:
            self.logger.error(f"[{request_id}] Error during streaming: {str(e)}")
            yield f"\n[Error: {str(e)}]"
            raise e

    def _handle_non_streaming(
        self, request_id: str, url: str, headers: dict, payload: dict
    ) -> str:
        """Handle non-streaming API responses"""
        self.debug(f"[{request_id}] Starting non-streaming request")

        start_time = time.time()
        response = self.client.post(url, headers=headers, json=payload)
        duration = time.time() - start_time

        # Raise for HTTP errors
        response.raise_for_status()

        # Parse response
        try:
            data = response.json()
            self.logger.info(
                f"[{request_id}] Non-streaming request completed in {duration:.2f}s"
            )

            # Extract text content
            if "content" in data and data["content"]:
                for content in data["content"]:
                    if content["type"] == "text":
                        return content["text"]

            # If we didn't find text content
            self.logger.warning(f"[{request_id}] No text content found in response")
            return ""

        except json.JSONDecodeError:
            self.logger.error(f"[{request_id}] Failed to parse JSON response")
            raise ValueError("Failed to parse JSON response from API")

        except Exception as e:
            self.logger.error(f"[{request_id}] Error processing response: {str(e)}")
            raise


# Create alias for backward compatibility
Pipe = EnhancedPipe
