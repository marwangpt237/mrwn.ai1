#!/usr/bin/env python3
"""
mrwn.ai – Enhanced Pollinations-Telegram-Bot with Privacy Protections
- Refactored for maintainability, robustness, and clarity.
- pip install python-telegram-bot==20.7 aiohttp python-dotenv
- export TELEGRAM_BOT_TOKEN="123:ABC"
- export ADMIN_ID="your_user_id"
- python bot.py
"""
import os
import json
import asyncio
import logging
import aiohttp
import base64
import io
import re
import html
from collections import defaultdict, deque
from datetime import datetime, timedelta
from urllib.parse import quote_plus
from typing import List, Dict, Tuple, Optional, Any

from dotenv import load_dotenv
from telegram import Update, constants, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, MessageHandler, filters, ContextTypes, CallbackQueryHandler
)

# ---------- CONFIGURATION ----------
class Config:
    """Centralized configuration for the bot."""
    load_dotenv()
    BOT_TOKEN: Optional[str] = os.getenv("TELEGRAM_BOT_TOKEN")
    ADMIN_ID: int = int(os.getenv("ADMIN_ID", 0))
    PROMPT_FILE: str = "prompt.md"

    # API Endpoints
    POLLEN_OPENAI: str = "https://text.pollinations.ai/openai"
    POLLEN_IMAGE: str = "https://image.pollinations.ai/prompt/"

    # Bot Behavior
    MAX_HISTORY: int = 10
    MAX_IMG_SIZE_MB: int = 20
    TELEGRAM_MSG_LIMIT: int = 4096

    # Rate Limiting
    RATE_LIMIT_WINDOW_S: int = 60
    RATE_LIMIT_MAX_REQUESTS: int = 5

    # Caching
    MAX_CACHE_ITEMS: int = 500

    # Logging
    LOG_LEVEL: int = logging.INFO
    LOG_FORMAT: str = "[%(asctime)s] [%(levelname)s] %(message)s"

    # Code Language Aliases
    CODE_LANGUAGES: Dict[str, str] = {
        'python': 'Python', 'py': 'Python', 'js': 'JavaScript', 'javascript': 'JavaScript',
        'html': 'HTML', 'css': 'CSS', 'java': 'Java', 'cpp': 'C++', 'c': 'C',
        'php': 'PHP', 'ruby': 'Ruby', 'go': 'Go', 'rust': 'Rust', 'sql': 'SQL',
        'swift': 'Swift', 'kotlin': 'Kotlin', 'bash': 'Bash', 'shell': 'Shell',
        'json': 'JSON', 'xml': 'XML', 'yaml': 'YAML', 'md': 'Markdown', 'markdown': 'Markdown'
    }

# ---------- STATE MANAGEMENT ----------
class BotState:
    """Manages the bot's runtime state to avoid global variables."""
    def __init__(self):
        self.user_rate_limit = defaultdict(lambda: deque(maxlen=Config.RATE_LIMIT_MAX_REQUESTS))
        self.user_conversations = defaultdict(list)
        self.code_cache = {} # Using dict for easy key-based access

    def check_rate_limit(self, user_id: int) -> bool:
        """Checks if a user has exceeded the rate limit. Returns True if allowed."""
        now = datetime.now()
        timestamps = self.user_rate_limit[user_id]
        # Remove old timestamps
        while timestamps and now - timestamps[0] > timedelta(seconds=Config.RATE_LIMIT_WINDOW_S):
            timestamps.popleft()

        if len(timestamps) >= Config.RATE_LIMIT_MAX_REQUESTS:
            return False
        timestamps.append(now)
        return True

    def add_to_conversation(self, user_id: int, role: str, content: str):
        """Adds a message to a user's conversation history."""
        self.user_conversations[user_id].append({"role": role, "content": content})
        # Trim history if it exceeds the max length
        if len(self.user_conversations[user_id]) > Config.MAX_HISTORY:
            self.user_conversations[user_id] = self.user_conversations[user_id][-Config.MAX_HISTORY:]

    def clear_conversation(self, user_id: int):
        """Clears a user's conversation history."""
        self.user_conversations[user_id] = []

    def cache_code_snippet(self, cache_id: str, code: str):
        """Stores a code snippet in the cache."""
        # Evict oldest items if cache is full
        if len(self.code_cache) >= Config.MAX_CACHE_ITEMS:
            # Simple FIFO eviction
            oldest_key = next(iter(self.code_cache))
            del self.code_cache[oldest_key]
        self.code_cache[cache_id] = code

    def get_cached_code(self, cache_id: str) -> Optional[str]:
        """Retrieves a code snippet from the cache."""
        return self.code_cache.get(cache_id)

# Initialize global state and logger
logging.basicConfig(level=Config.LOG_LEVEL, format=Config.LOG_FORMAT)
bot_state = BotState()

# ---------- PROMPT LOADER ----------
def load_system_prompt() -> str:
    """Loads the system prompt from a file or returns a default."""
    try:
        with open(Config.PROMPT_FILE, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        logging.warning(f"'{Config.PROMPT_FILE}' not found. Using fallback system prompt.")
        return (
            "You are mrwn.ai, a helpful AI assistant. You have conversation memory and "
            "can remember the last 10 messages. Always format code with proper syntax "
            "highlighting. Maintain user privacy and refuse to generate harmful content."
        )

SYSTEM_PROMPT = load_system_prompt()

# ---------- UTILITIES ----------
async def download(url: str, params: Optional[dict] = None, post_json: Optional[dict] = None) -> bytes:
    """Performs an async GET or POST request and returns the response body."""
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120)) as session:
        request_type = "POST" if post_json else "GET"
        try:
            if post_json:
                async with session.post(url, json=post_json) as response:
                    response.raise_for_status()
                    return await response.read()
            else:
                async with session.get(url, params=params) as response:
                    response.raise_for_status()
                    return await response.read()
        except aiohttp.ClientResponseError as e:
            logging.error(f"{request_type} to {url} failed with status {e.status}: {e.message}")
            raise  # Re-raise to be caught by the handler
        except asyncio.TimeoutError:
            logging.error(f"{request_type} to {url} timed out.")
            raise
        except aiohttp.ClientError as e:
            logging.error(f"AIOHTTP client error during request to {url}: {e}")
            raise

def sanitize_for_logging(text: str) -> str:
    """Removes potentially sensitive information from text before logging."""
    patterns = [
        re.compile(r'sk-[A-Za-z0-9]{20,}'),          # OpenAI-like keys
        re.compile(r'ghp_[A-Za-z0-9]{20,}'),         # GitHub PATs
        re.compile(r'AIza[0-9A-Za-z-_]{35}'),        # Google API-like keys
        re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'), # Emails
        re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b'),   # IPv4 addresses
    ]
    for p in patterns:
        text = p.sub('[REDACTED]', text)
    return text

async def handle_api_error(update: Update, context: ContextTypes.DEFAULT_TYPE, error: Exception, service: str):
    """Logs detailed errors and sends a user-friendly message."""
    user_id = update.effective_user.id if update.effective_user else "N/A"
    logging.error(f"Error in {service} for user {user_id}: {sanitize_for_logging(str(error))}")

    if isinstance(error, aiohttp.ClientResponseError):
        if 400 <= error.status < 500:
            msg = f"⚠️ A client-side error occurred ({error.status}). Please check your input and try again."
        elif 500 <= error.status < 600:
            msg = f"🔧 The {service} service is temporarily unavailable ({error.status}). Please try again later."
        else:
            msg = "An unexpected network error occurred. Please try again."
    elif isinstance(error, asyncio.TimeoutError):
        msg = "⏳ The request timed out. The service may be busy. Please try again later."
    else:
        msg = "⚠️ An unexpected error occurred. The issue has been logged."

    try:
        if update.callback_query:
            await update.callback_query.message.reply_text(msg)
        elif update.message:
            await update.message.reply_text(msg)
    except Exception as e:
        logging.error(f"Failed to send error message to user {user_id}: {e}")

# ---------- MARKDOWN TO HTML RENDERING ----------
CODE_BLOCK_RE = re.compile(r"```(\w*)\n(.*?)\n```", re.DOTALL)
INLINE_CODE_RE = re.compile(r"`([^`\n]+?)`")

def render_markdown_to_html(md_text: str, message_id: int) -> Tuple[List[str], Optional[InlineKeyboardMarkup]]:
    """
    Converts markdown with code blocks to a list of safe HTML message chunks
    and an InlineKeyboardMarkup with copy buttons.
    """
    chunks = []
    current_chunk = ""
    keyboard_buttons = []
    code_index = 0
    last_pos = 0

    for match in CODE_BLOCK_RE.finditer(md_text):
        # 1. Add preceding text segment to the current chunk
        preceding_text = md_text[last_pos:match.start()]
        if preceding_text.strip():
            escaped_text = html.escape(preceding_text)
            escaped_text_with_inline = INLINE_CODE_RE.sub(r"<code>\1</code>", escaped_text)
            current_chunk += escaped_text_with_inline

        # 2. Process the code block
        lang, code = match.groups()
        lang = lang.strip().lower()
        code = code.strip()

        cache_id = f"{message_id}_{code_index}"
        bot_state.cache_code_snippet(cache_id, code)

        lang_display = Config.CODE_LANGUAGES.get(lang, lang.capitalize() if lang else "Code")
        btn_text = f"📋 Copy {lang_display} ({code_index + 1})"
        keyboard_buttons.append([InlineKeyboardButton(btn_text, callback_data=f"copy_{cache_id}")])

        code_html = f"<pre><code>{html.escape(code)}</code></pre>"

        # 3. Add code block to chunks, splitting if necessary
        if len(current_chunk) + len(code_html) > Config.TELEGRAM_MSG_LIMIT:
            if current_chunk:
                chunks.append(current_chunk)
            chunks.append(code_html)
            current_chunk = ""
        else:
            current_chunk += code_html

        last_pos = match.end()
        code_index += 1

    # 4. Add any remaining text after the last code block
    remaining_text = md_text[last_pos:]
    if remaining_text.strip():
        escaped_text = html.escape(remaining_text)
        escaped_text_with_inline = INLINE_CODE_RE.sub(r"<code>\1</code>", escaped_text)
        if len(current_chunk) + len(escaped_text_with_inline) > Config.TELEGRAM_MSG_LIMIT:
            chunks.append(current_chunk)
            current_chunk = escaped_text_with_inline
        else:
            current_chunk += escaped_text_with_inline

    if current_chunk:
        chunks.append(current_chunk)

    reply_markup = InlineKeyboardMarkup(keyboard_buttons) if keyboard_buttons else None
    return chunks or [""], reply_markup


# ---------- TELEGRAM HANDLERS ----------
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_html(
        "🤖 <b>mrwn.ai is online!</b>\n\n"
        "I'm an AI assistant with conversation memory, image generation, and vision capabilities.\n\n"
        "<b>Commands:</b>\n"
        "/help - Show this guide\n"
        "/img &lt;prompt&gt; - Create an image\n"
        "/clear - Reset our conversation\n\n"
        "Simply send me a message or a photo to get started!"
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_html(
        "<b>mrwn.ai Help Guide</b>\n\n"
        "<b>Chat:</b> Just send any text message. I remember the last 10 messages.\n\n"
        "<b>Image Generation:</b>\n"
        "• <code>/img a robot cat</code>\n"
        "• <code>/img 512x1024 a robot cat</code> (Custom size from 256 to 2048px)\n\n"
        "<b>Vision:</b> Send a photo with a caption. If you don't add a caption, I'll describe the image.\n\n"
        "<b>Code:</b> I automatically format code and provide copy buttons for easy use.\n\n"
        "<b>Memory:</b> Use <code>/clear</code> to start a fresh conversation."
    )

async def clear_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    bot_state.clear_conversation(user_id)
    await update.message.reply_text("🧹 Conversation history cleared.")

async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != Config.ADMIN_ID:
        return
    active_convos = sum(1 for v in bot_state.user_conversations.values() if v)
    stats_text = (
        f"📊 <b>Bot Statistics</b>\n"
        f"• Tracked Users: {len(bot_state.user_conversations)}\n"
        f"• Active Conversations: {active_convos}\n"
        f"• Cached Code Snippets: {len(bot_state.code_cache)}"
    )
    await update.message.reply_html(stats_text)

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    if query.data and query.data.startswith("copy_"):
        cache_id = query.data[len("copy_"):]
        code = bot_state.get_cached_code(cache_id)
        if code:
            # Truncate if needed for the reply message
            if len(code) > 3500:
                code_to_send = code[:3500] + "\n\n... (truncated for display)"
            else:
                code_to_send = code
            await query.message.reply_html(f"📋 <b>Code:</b>\n<pre><code>{html.escape(code_to_send)}</code></pre>")
        else:
            await context.bot.answer_callback_query(query.id, "Code snippet has expired from cache.", show_alert=True)

async def text_chat_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_text = update.message.text
    logging.info(f"User {user_id} (chat): {sanitize_for_logging(user_text)}")

    if not bot_state.check_rate_limit(user_id):
        await update.message.reply_text("🚦 Rate limit exceeded. Please wait a moment.")
        return

    status_msg = await update.message.reply_text("🤔 Thinking...")
    await context.bot.send_chat_action(update.effective_chat.id, constants.ChatAction.TYPING)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(bot_state.user_conversations.get(user_id, []))
    messages.append({"role": "user", "content": user_text})

    payload = {"model": "openai", "messages": messages, "private": True}

    try:
        resp_bytes = await download(Config.POLLEN_OPENAI, post_json=payload)
        data = json.loads(resp_bytes)
        reply_text = data["choices"][0]["message"]["content"]

        bot_state.add_to_conversation(user_id, "user", user_text)
        bot_state.add_to_conversation(user_id, "assistant", reply_text)

        await status_msg.delete()
        chunks, reply_markup = render_markdown_to_html(reply_text, update.message.message_id)
        for i, chunk in enumerate(chunks):
            await update.message.reply_html(chunk, reply_markup=(reply_markup if i == len(chunks) - 1 else None))

    except Exception as e:
        await status_msg.delete()
        await handle_api_error(update, context, e, "Chat")

async def image_generation_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not bot_state.check_rate_limit(user_id):
        await update.message.reply_text("🚦 Rate limit exceeded. Please wait a moment.")
        return

    if not context.args:
        await update.message.reply_html("<b>Usage:</b> <code>/img [width]x[height] <prompt></code>")
        return

    prompt_parts = context.args
    width, height = 1024, 1024
    match = re.match(r"(\d+)[xX](\d+)", prompt_parts[0])
    if match:
        width, height = int(match.group(1)), int(match.group(2))
        prompt = " ".join(prompt_parts[1:])
    else:
        prompt = " ".join(prompt_parts)

    width = max(256, min(width, 2048))
    height = max(256, min(height, 2048))

    if not prompt:
        await update.message.reply_text("Please provide a prompt after the command.")
        return

    logging.info(f"User {user_id} (image): {sanitize_for_logging(prompt)}")
    status_msg = await update.message.reply_text("🎨 Creating your image...")
    await context.bot.send_chat_action(update.effective_chat.id, constants.ChatAction.UPLOAD_PHOTO)

    url = f"{Config.POLLEN_IMAGE}{quote_plus(prompt)}?width={width}&height={height}&model=flux&private=true"

    try:
        img_bytes = await download(url)
        caption = f"🖼️ {prompt}\nSize: {width}x{height}"
        await update.message.reply_photo(photo=img_bytes, caption=caption[:1024])
        await status_msg.delete()
    except Exception as e:
        await status_msg.delete()
        await handle_api_error(update, context, e, "Image Generation")


async def vision_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not bot_state.check_rate_limit(user_id):
        await update.message.reply_text("🚦 Rate limit exceeded. Please wait a moment.")
        return

    photo_file = await update.message.photo[-1].get_file()
    img_buffer = io.BytesIO()
    await photo_file.download_to_memory(img_buffer)
    img_buffer.seek(0)

    if img_buffer.getbuffer().nbytes > Config.MAX_IMG_SIZE_MB * 1024 * 1024:
        await update.message.reply_text(f"❌ Image is too large. Max size is {Config.MAX_IMG_SIZE_MB}MB.")
        return

    status_msg = await update.message.reply_text("👀 Analyzing image...")
    await context.bot.send_chat_action(update.effective_chat.id, constants.ChatAction.TYPING)

    b64_img = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
    prompt_text = update.message.caption or "Describe this image in detail."

    logging.info(f"User {user_id} (vision): {sanitize_for_logging(prompt_text)}")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(bot_state.user_conversations.get(user_id, []))
    messages.append({
        "role": "user",
        "content": [
            {"type": "text", "text": prompt_text},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}
        ]
    })

    payload = {"model": "openai", "messages": messages, "max_tokens": 1024, "private": True}

    try:
        resp_bytes = await download(Config.POLLEN_OPENAI, post_json=payload)
        data = json.loads(resp_bytes)
        reply_text = data["choices"][0]["message"]["content"]

        # Add a placeholder for the image in history
        bot_state.add_to_conversation(user_id, "user", f"[Image Attached] {prompt_text}")
        bot_state.add_to_conversation(user_id, "assistant", reply_text)

        await status_msg.delete()
        chunks, reply_markup = render_markdown_to_html(reply_text, update.message.message_id)
        for i, chunk in enumerate(chunks):
            await update.message.reply_html(chunk, reply_markup=(reply_markup if i == len(chunks) - 1 else None))

    except Exception as e:
        await status_msg.delete()
        await handle_api_error(update, context, e, "Vision")

# ---------- MAIN ----------
def main():
    """Sets up and runs the Telegram bot."""
    if not Config.BOT_TOKEN:
        logging.critical("FATAL: TELEGRAM_BOT_TOKEN environment variable not set.")
        return

    app = Application.builder().token(Config.BOT_TOKEN).build()

    # Command Handlers
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("clear", clear_command))
    app.add_handler(CommandHandler("stats", stats_command))
    app.add_handler(CommandHandler("img", image_generation_handler))

    # Message Handlers
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_chat_handler))
    app.add_handler(MessageHandler(filters.PHOTO, vision_handler))

    # Callback Handler for buttons
    app.add_handler(CallbackQueryHandler(button_callback))

    logging.info("Bot is starting...")
    app.run_polling()

if __name__ == "__main__":
    main()
