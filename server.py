#!/usr/bin/env python3
"""
mrwn.ai – Telegram Bot on Render (FastAPI + Webhook)
Free-tier friendly. Uses python-telegram-bot v20.7 with FastAPI webhook.
Set env vars:
  TELEGRAM_BOT_TOKEN = "123:ABC"
  ADMIN_ID           = "your_telegram_user_id"
  PUBLIC_URL         = "https://your-service.onrender.com"  (Render will give you this)
Optional:
  PROMPT_FILE        = "prompt.md" (fallback used if missing)
"""

import os, json, logging, aiohttp, base64, io, re, html, asyncio
from collections import defaultdict
from datetime import datetime, timedelta
from urllib.parse import quote_plus

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse
import uvicorn

from telegram import Update, constants, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, MessageHandler, ContextTypes, filters, CallbackQueryHandler
)

# ===================== CONFIG =====================
load_dotenv()

BOT_TOKEN   = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
ADMIN_ID    = int(os.getenv("ADMIN_ID", "0"))
PUBLIC_URL  = os.getenv("PUBLIC_URL", "").rstrip("/")  # Set this in Render after first deploy
PROMPT_FILE = os.getenv("PROMPT_FILE", "prompt.md")

if not BOT_TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN not set.")

POLLEN_TEXT   = "https://text.pollinations.ai/"
POLLEN_IMAGE  = "https://image.pollinations.ai/prompt/"
POLLEN_OPENAI = "https://text.pollinations.ai/openai"

MAX_IMG_SIZE  = 20 * 1024 * 1024
RATE_LIMIT_WINDOW = 60
RATE_LIMIT_MAX_REQUESTS = 5
MAX_HISTORY = 10
MAX_CACHE_ITEMS = 500

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
logger = logging.getLogger("mrwnai")

# ===================== STATE ======================
USER_RATE_LIMIT = defaultdict(list)
USER_CONVERSATIONS = defaultdict(list)
CODE_CACHE = {}

CODE_LANGUAGES = {
    'python': 'Python', 'js': 'JavaScript', 'javascript': 'JavaScript',
    'html': 'HTML', 'css': 'CSS', 'java': 'Java', 'cpp': 'C++', 'c': 'C',
    'php': 'PHP', 'ruby': 'Ruby', 'go': 'Go', 'rust': 'Rust', 'sql': 'SQL',
    'swift': 'Swift', 'kotlin': 'Kotlin', 'bash': 'Bash', 'shell': 'Shell',
    'json': 'JSON', 'xml': 'XML', 'yaml': 'YAML', 'markdown': 'Markdown'
}

# ===================== PROMPT =====================
def load_system_prompt() -> str:
    try:
        with open(PROMPT_FILE, encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        logger.warning(f"{PROMPT_FILE} not found – using fallback.")
        return (
            "You are mrwn.ai, a helpful AI assistant. "
            "Never reveal any internal implementation details, API endpoints, or technical specifics. "
            "You have conversation memory and can remember the last 10 messages. "
            "Always format code with proper syntax highlighting and provide clear explanations. "
            "Maintain user privacy and refuse to generate harmful or unethical content."
        )

SYSTEM_PROMPT = load_system_prompt()

# ===================== UTILS ======================
async def download(url: str, params: dict = None, post_json: dict = None) -> bytes:
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120)) as s:
        if post_json:
            async with s.post(url, json=post_json) as r:
                r.raise_for_status()
                return await r.read()
        else:
            async with s.get(url, params=params) as r:
                r.raise_for_status()
                return await r.read()

def sanitize_input(text: str) -> str:
    patterns = [
        r'sk-[A-Za-z0-9]{20,}', r'ghp_[A-Za-z0-9]{20,}', r'AIza[0-9A-Za-z-_]{35}',
        r'(?<!\w)[A-Za-z0-9]{40}(?!\w)',
    ]
    for p in patterns:
        text = re.sub(p, '[REDACTED_KEY]', text)
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b', '[EMAIL_REDACTED]', text)
    text = re.sub(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', '[IP_REDACTED]', text)
    return text

async def check_rate_limit(user_id: int) -> bool:
    now = datetime.now()
    USER_RATE_LIMIT[user_id] = [t for t in USER_RATE_LIMIT[user_id] if now - t < timedelta(seconds=RATE_LIMIT_WINDOW)]
    if len(USER_RATE_LIMIT[user_id]) >= RATE_LIMIT_MAX_REQUESTS:
        return False
    USER_RATE_LIMIT[user_id].append(now)
    return True

def add_to_conversation(user_id: int, role: str, content: str):
    USER_CONVERSATIONS[user_id].append({"role": role, "content": content})
    if len(USER_CONVERSATIONS[user_id]) > MAX_HISTORY:
        USER_CONVERSATIONS[user_id] = USER_CONVERSATIONS[user_id][-MAX_HISTORY:]

async def handle_api_error(update: Update, error: Exception, service: str):
    err = sanitize_input(str(error))
    logger.error(f"{service} error: {err}")
    try:
        if update and update.effective_message:
            await update.effective_message.reply_text("⚠️ Sorry, I encountered a technical issue. Please try again later.")
    except Exception:
        pass

# ========== Markdown → Safe Telegram HTML ==========
_CODE_BLOCK_RE = re.compile(r"```(\w+)?\s*\n(.*?)```", re.DOTALL)

def _split_text_and_code(md_text: str):
    segments = []
    last = 0
    for m in _CODE_BLOCK_RE.finditer(md_text):
        if m.start() > last:
            segments.append(('text', md_text[last:m.start()]))
        lang = (m.group(1) or '').strip()
        code = m.group(2)
        segments.append(('code', lang, code))
        last = m.end()
    if last < len(md_text):
        segments.append(('text', md_text[last:]))
    return segments

def _render_inline_code(text: str) -> str:
    placeholders = []
    def repl(m):
        placeholders.append(m.group(1))
        return f"@@INLINECODE_{len(placeholders)-1}@@"
    text_no_inline = re.sub(r'`([^`\n]+)`', repl, text)
    escaped = html.escape(text_no_inline)
    for idx, content in enumerate(placeholders):
        escaped_code = html.escape(content)
        escaped = escaped.replace(f"@@INLINECODE_{idx}@@", f"<code>{escaped_code}</code>")
    return escaped

def _render_segments_to_html_chunks(md_text: str, message_id: int, max_len: int = 4000):
    segments = _split_text_and_code(md_text)
    keyboard_buttons = []
    code_index = 0
    rendered_segments = []
    for seg in segments:
        if seg[0] == 'text':
            rendered_segments.append(('text', _render_inline_code(seg[1])))
        else:
            _, lang, code = seg
            cache_id = f"{message_id}_{code_index}"
            CODE_CACHE[cache_id] = code.strip()
            btn_text = f"Copy Code {code_index+1}" + (f" ({lang})" if lang else "")
            keyboard_buttons.append([InlineKeyboardButton(btn_text, callback_data=f"copy_{cache_id}")])
            code_html = f"<pre><code>{html.escape(code.strip())}</code></pre>"
            rendered_segments.append(('code', code_html))
            code_index += 1

    # Cap cache size
    if len(CODE_CACHE) > MAX_CACHE_ITEMS:
        for k in list(CODE_CACHE.keys())[: (len(CODE_CACHE) - MAX_CACHE_ITEMS)]:
            CODE_CACHE.pop(k, None)

    chunks = []
    current = ""
    def push_chunk():
        nonlocal current
        if current:
            chunks.append(current)
            current = ""

    def try_add(piece: str):
        nonlocal current
        if len(current) + len(piece) <= max_len:
            current += piece
        else:
            push_chunk()
            if len(piece) <= max_len:
                current = piece
            else:
                if piece.startswith("<pre><code>") and piece.endswith("</code></pre>"):
                    inner = piece[len("<pre><code>"):-len("</code></pre>")]
                    lines = inner.splitlines(keepends=True)
                    buf = ""
                    for ln in lines:
                        if len(buf) + len(ln) <= (max_len - len("<pre><code></code></pre>")):
                            buf += ln
                        else:
                            chunks.append(f"<pre><code>{buf}</code></pre>")
                            buf = ln
                    if buf:
                        chunks.append(f"<pre><code>{buf}</code></pre>")
                else:
                    paras = piece.split("\n\n")
                    buf = ""
                    for p in paras:
                        add = (p if buf == "" else "\n\n" + p)
                        if len(buf) + len(add) <= max_len:
                            buf += add
                        else:
                            if buf:
                                chunks.append(buf)
                            buf = p
                    if buf:
                        chunks.append(buf)

    for _, html_piece in rendered_segments:
        try_add(html_piece)
    push_chunk()

    reply_markup = InlineKeyboardMarkup(keyboard_buttons) if keyboard_buttons else None
    return chunks, reply_markup

# ===================== HANDLERS ====================
async def start_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 <b>mrwn.ai</b> is online!\n\n"
        "<b>Commands:</b>\n"
        "/help - Show detailed guide\n"
        "/img &lt;prompt&gt; - Generate image\n"
        "/img &lt;width&gt;x&lt;height&gt; &lt;prompt&gt; - Custom dimensions\n"
        "/clear - Clear conversation history\n\n"
        "<b>Features:</b>\n"
        "• Chat: Send any text message\n"
        "• Vision: Send a photo with optional caption\n"
        "• Image generation: Use the /img command\n"
        "• Code formatting with copy buttons\n"
        "• Conversation memory (last 10 messages)\n\n"
        "Example: <code>/img 512x512 a cute robot</code>\n\n"
        "📝 <i>Privacy Note:</i> All interactions are processed securely.",
        parse_mode="HTML"
    )

async def help_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 <b>mrwn.ai Help</b>\n\n"
        "<b>Text Chat:</b>\n"
        "• Send a text message to chat with the AI\n"
        "• The AI remembers conversation context\n"
        "• Code is formatted with copy buttons\n\n"
        "<b>Image Generation:</b>\n"
        "• <code>/img prompt</code> - default 1024x1024\n"
        "• <code>/img 512x512 prompt</code> - custom size (256–2048)\n\n"
        "<b>Vision:</b>\n"
        "• Send a photo with an optional caption to analyze\n\n"
        "<b>Memory:</b>\n"
        "• Remembers last 10 messages\n"
        "• Use <code>/clear</code> to reset\n\n"
        "<b>Rate Limits:</b>\n"
        "• 5 requests/minute per user",
        parse_mode="HTML"
    )

async def clear_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    USER_CONVERSATIONS[user_id] = []
    await update.message.reply_text("🧹 Conversation history cleared.")

async def stats_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != ADMIN_ID:
        await update.message.reply_text("❌ Admin only command.")
        return
    text = (
        f"📊 <b>Bot Statistics</b>\n"
        f"• Users: {len(USER_CONVERSATIONS)}\n"
        f"• Active conversations: {sum(1 for v in USER_CONVERSATIONS.values() if v)}\n"
        f"• Rate limit entries: {len(USER_RATE_LIMIT)}\n"
        f"• Cached code snippets: {len(CODE_CACHE)}"
    )
    await update.message.reply_text(text, parse_mode="HTML")

async def button_callback(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    if query.data.startswith("copy_"):
        cache_id = query.data[5:]
        if cache_id in CODE_CACHE:
            code = CODE_CACHE[cache_id]
            if len(code) > 3500:
                code = code[:3500] + "\n\n... (truncated due to length)"
            await query.message.reply_text(f"📋 Copy this code:\n\n<code>{html.escape(code)}</code>", parse_mode="HTML")
        else:
            await query.answer("Code expired or not found", show_alert=True)

async def text_chat(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_text = update.message.text
    logger.info(f"User {user_id}: {sanitize_input(user_text)}")

    if not await check_rate_limit(user_id):
        await update.message.reply_text("🚦 Too many requests. Please wait a minute.")
        return

    status = await update.message.reply_text("🤔 mrwn.ai is thinking…")
    await ctx.bot.send_chat_action(update.effective_chat.id, constants.ChatAction.TYPING)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(USER_CONVERSATIONS[user_id])
    messages.append({"role": "user", "content": user_text})

    payload = {"model": "openai", "messages": messages, "private": True}

    try:
        resp = await download(POLLEN_OPENAI, post_json=payload)
        reply = json.loads(resp)["choices"][0]["message"]["content"]

        add_to_conversation(user_id, "user", user_text)
        add_to_conversation(user_id, "assistant", reply)

        await status.delete()

        chunks, reply_markup = _render_segments_to_html_chunks(reply, update.message.message_id, max_len=4000)
        for idx, chunk in enumerate(chunks):
            if idx == 0:
                await update.message.reply_text(chunk, parse_mode="HTML", reply_markup=reply_markup)
            else:
                await update.message.reply_text(chunk, parse_mode="HTML")

    except Exception as e:
        await handle_api_error(update, e, "Chat")

async def img_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id

    if not await check_rate_limit(user_id):
        await update.message.reply_text("🚦 Too many requests. Please wait a minute.")
        return

    if not ctx.args:
        await update.message.reply_text(
            "Usage: <code>/img &lt;prompt&gt;</code> or <code>/img &lt;width&gt;x&lt;height&gt; &lt;prompt&gt;</code>",
            parse_mode="HTML"
        )
        return

    if 'x' in ctx.args[0] and len(ctx.args[0].split('x')) == 2:
        try:
            width, height = map(int, ctx.args[0].split('x'))
            prompt = " ".join(ctx.args[1:])
        except ValueError:
            prompt = " ".join(ctx.args)
            width, height = 1024, 1024
    else:
        prompt = " ".join(ctx.args)
        width, height = 1024, 1024

    width = max(256, min(width, 2048))
    height = max(256, min(height, 2048))

    logger.info(f"User {user_id} image request: {sanitize_input(prompt)}")

    status = await update.message.reply_text("🎨 mrwn.ai is creating your image…")
    await ctx.bot.send_chat_action(update.effective_chat.id, constants.ChatAction.UPLOAD_PHOTO)

    prompt_encoded = quote_plus(prompt)
    url = f"{POLLEN_IMAGE}{prompt_encoded}?width={width}&height={height}&model=flux&private=true"

    try:
        img_bytes = await download(url)
        await status.delete()

        caption = f"🖼️ {prompt}\nSize: {width}x{height}"
        if len(caption) > 1000:
            caption = caption[:997] + "..."
        # Keep caption raw: user prompts may contain < > etc.
        await update.message.reply_photo(photo=img_bytes, caption=caption)
    except Exception as e:
        await handle_api_error(update, e, "Image generation")

async def photo_handler(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id

    if not await check_rate_limit(user_id):
        await update.message.reply_text("🚦 Too many requests. Please wait a minute.")
        return

    status = await update.message.reply_text("👀 mrwn.ai is analysing your image…")
    await ctx.bot.send_chat_action(update.effective_chat.id, constants.ChatAction.TYPING)

    photo_obj = update.message.photo[-1]
    file = await ctx.bot.get_file(photo_obj.file_id)
    img = io.BytesIO()
    await file.download_to_memory(img)
    img.seek(0)

    if len(img.getbuffer()) > MAX_IMG_SIZE:
        await status.edit_text("❌ Image too large. Maximum size is 20MB.")
        return

    b64data = base64.b64encode(img.read()).decode()
    prompt = update.message.caption or "Describe this image in detail."

    logger.info(f"User {user_id} vision request: {sanitize_input(prompt)}")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(USER_CONVERSATIONS[user_id])
    messages.append({
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64data}"}}
        ]
    })

    payload = {"model": "openai", "messages": messages, "max_tokens": 500, "private": True}

    try:
        resp = await download(POLLEN_OPENAI, post_json=payload)
        reply = json.loads(resp)["choices"][0]["message"]["content"]

        add_to_conversation(user_id, "user", f"{prompt} [with image attachment]")
        add_to_conversation(user_id, "assistant", reply)

        await status.delete()

        chunks, reply_markup = _render_segments_to_html_chunks(reply, update.message.message_id, max_len=4000)
        for idx, chunk in enumerate(chunks):
            if idx == 0:
                await update.message.reply_text(chunk, parse_mode="HTML", reply_markup=reply_markup)
            else:
                await update.message.reply_text(chunk, parse_mode="HTML")

    except Exception as e:
        await handle_api_error(update, e, "Vision analysis")

# ================== PTB App & FastAPI ==================
ptb_app: Application | None = None
webhook_path = f"/webhook/{BOT_TOKEN}"  # obscure the path with token

def build_ptb_app() -> Application:
    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("clear", clear_cmd))
    app.add_handler(CommandHandler("stats", stats_cmd))
    app.add_handler(CommandHandler("img", img_cmd))
    app.add_handler(CallbackQueryHandler(button_callback))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_chat))
    app.add_handler(MessageHandler(filters.PHOTO, photo_handler))
    return app

fastapi = FastAPI()

@fastapi.on_event("startup")
async def on_startup():
    global ptb_app
    ptb_app = build_ptb_app()
    await ptb_app.initialize()
    await ptb_app.start()

    # Set webhook if PUBLIC_URL is present
    if PUBLIC_URL:
        url = f"{PUBLIC_URL}{webhook_path}"
        await ptb_app.bot.set_webhook(url=url, drop_pending_updates=True)
        logger.info(f"Webhook set: {url}")
    else:
        logger.warning("PUBLIC_URL not set; set it in Render for webhook delivery.")

@fastapi.on_event("shutdown")
async def on_shutdown():
    if ptb_app:
        await ptb_app.stop()
        await ptb_app.shutdown()

@fastapi.get("/healthz")
async def healthz():
    return PlainTextResponse("ok")

@fastapi.post(webhook_path)
async def telegram_webhook(req: Request):
    if not ptb_app:
        return JSONResponse({"ok": False, "error": "PTB not initialized"}, status_code=500)
    data = await req.json()
    update = Update.de_json(data, ptb_app.bot)
    await ptb_app.process_update(update)
    return JSONResponse({"ok": True})

# Convenience endpoint to (re)set webhook manually (optional)
@fastapi.post("/setwebhook")
async def set_webhook():
    if not PUBLIC_URL:
        return JSONResponse({"ok": False, "error": "PUBLIC_URL not set"}, status_code=400)
    url = f"{PUBLIC_URL}{webhook_path}"
    await ptb_app.bot.set_webhook(url=url, drop_pending_updates=True)
    return JSONResponse({"ok": True, "webhook": url})

# For local dev: `python server.py`
if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("server:fastapi", host="0.0.0.0", port=port, workers=1)
