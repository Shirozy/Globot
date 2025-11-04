import discord
from discord import app_commands
from discord.ext import commands
import asyncio, aiohttp, json, os, re
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from collections import Counter
import torch

# -------------------- Configuration --------------------
load_dotenv()
TOKEN = os.getenv("DISCORD_TOKEN")
BOT_OWNER_ID = int(os.getenv("BOT_OWNER_ID", "0"))
DEBUG_LEVEL = os.getenv("DEBUG_LEVEL", "none").lower()  # none, error, mod, all

ENABLE_TOXICITY = os.getenv("ENABLE_TOXICITY", "True").lower() == "true"
ENABLE_TRANSLATION = os.getenv("ENABLE_TRANSLATION", "True").lower() == "true"
TRANSLATE_URL = os.getenv("LIBRETRANSLATE_URL", "http://localhost:5000/translate")

INTENTS = discord.Intents.default()
INTENTS.message_content = True
INTENTS.guilds = True
INTENTS.messages = True

bot = commands.Bot(command_prefix="!", intents=INTENTS)

# -------------------- Debugging --------------------
def debug_log(level: str, message: str):
    levels = {"none": 0, "error": 1, "mod": 2, "all": 3}
    if levels.get(DEBUG_LEVEL, 0) >= levels.get(level, 0):
        print(f"[{level.upper()}] {message}")

# -------------------- Data Handling --------------------
if not os.path.exists("data.json"):
    with open("data.json", "w") as f:
        json.dump({"channels": {}, "logs": {}, "warnings": {}, "webhooks": {}}, f)

def load_data():
    with open("data.json", "r") as f:
        return json.load(f)

def save_data(data):
    with open("data.json", "w") as f:
        json.dump(data, f, indent=2)

data = load_data()

# -------------------- Toxicity Model --------------------
if ENABLE_TOXICITY:
    MODEL_NAME = "unitary/toxic-bert"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

    def classify_text(text: str):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        outputs = model(**inputs)
        probs = torch.sigmoid(outputs.logits)[0].tolist()
        return dict(zip(labels, probs))

# -------------------- Permissions Checker --------------------
REQUIRED_PERMS = [
    "manage_webhooks",
    "manage_channels",
    "manage_messages",
    "read_messages",
    "send_messages",
    "read_message_history",
    "embed_links",
    "use_application_commands",
]

async def check_permissions(guild: discord.Guild):
    """Check for missing permissions in all text channels."""
    missing = {}
    me = guild.me
    for channel in guild.text_channels:
        perms = channel.permissions_for(me)
        missing_in_channel = [p for p in REQUIRED_PERMS if not getattr(perms, p, False)]
        if missing_in_channel:
            missing[channel.name] = missing_in_channel
    if missing:
        debug_log("error", f"Missing permissions in {guild.name}: {missing}")
    return missing

# -------------------- Helper Functions --------------------
async def translate_text(text: str, target_lang: str) -> str:
    if not ENABLE_TRANSLATION:
        return text
    async with aiohttp.ClientSession() as session:
        payload = {"q": text, "source": "auto", "target": target_lang}
        try:
            async with session.post(TRANSLATE_URL, json=payload) as r:
                if r.status == 200:
                    js = await r.json()
                    return js.get("translatedText", text)
        except Exception as e:
            debug_log("error", f"Translation error: {e}")
            return text
    return text

async def is_toxic(text: str):
    if not ENABLE_TOXICITY:
        return False, {}
    scores = await asyncio.to_thread(classify_text, text)
    return any(v > 0.5 for v in scores.values()), scores

async def add_warning(guild_id: int, user_id: int):
    warnings = data["warnings"].setdefault(str(guild_id), {})
    warnings[str(user_id)] = warnings.get(str(user_id), 0) + 1
    save_data(data)

def get_warnings(guild_id: int, user_id: int) -> int:
    return data["warnings"].get(str(guild_id), {}).get(str(user_id), 0)

async def log_toxic_message(guild: discord.Guild, user: discord.User, message: discord.Message, scores: dict):
    log_channel_id = data["logs"].get(str(guild.id))
    if not log_channel_id:
        return
    log_channel = guild.get_channel(log_channel_id)
    if log_channel:
        embed = discord.Embed(title="Toxic Message Detected", color=discord.Color.red())
        embed.add_field(name="User", value=f"{user.mention}", inline=True)
        embed.add_field(name="Message", value=message.content or "[No content]", inline=False)
        embed.add_field(name="Scores", value="\n".join([f"{k}: {v:.2f}" for k, v in scores.items()]))
        await log_channel.send(embed=embed)
        debug_log("mod", f"Toxic message by {user} removed in {guild.name}")

# -------------------- Webhook Sending --------------------
async def send_to_channel(channel: discord.TextChannel, author: discord.User, content: str):
    if re.search(r"https?://", content):
        debug_log("mod", f"Skipped link message from {author} in {channel.guild.name}")
        return
    try:
        wh_id = data["webhooks"].get(str(channel.id))
        webhook = None
        if wh_id:
            try:
                webhook = await bot.fetch_webhook(wh_id)
            except discord.NotFound:
                webhook = None
        if not webhook:
            webhook = await channel.create_webhook(name="SyncBot")
            data["webhooks"][str(channel.id)] = webhook.id
            save_data(data)
        await webhook.send(
            content=content,
            username=author.display_name[:32],
            avatar_url=author.display_avatar.url,
            allowed_mentions=discord.AllowedMentions.none()
        )
        debug_log("mod", f"Forwarded message from {author} to {channel.guild.name}")
    except discord.Forbidden:
        debug_log("error", f"Missing webhook permissions in {channel.name}")
    except Exception as e:
        debug_log("error", f"Webhook error: {e}")

# -------------------- Slash Commands --------------------
@bot.tree.command(name="addchannel", description="Add this channel to the cross-server sync and set its language.")
@app_commands.describe(lang="Language code (e.g., en, es, fr)")
async def add_channel(interaction: discord.Interaction, lang: str):
    await interaction.response.defer(ephemeral=True)

    embed = discord.Embed(
        title="Terms of Service for Cross-Server Sync Bot",
        description=(
            "By using this bot in your channel, you agree to the following terms:\n\n"
            "**Content Restrictions:**\n"
            "- NSFW, hateful, or illegal content is prohibited.\n"
            "- Toxic messages may be automatically deleted.\n\n"
            "**Message Handling:**\n"
            "- Messages may be forwarded to other servers.\n"
            "- Once sent, messages cannot be deleted or edited.\n\n"
            "**Moderation:**\n"
            "- Toxic messages will trigger warnings and may be logged.\n"
            "- Repeated violations may result in removal from the sync system.\n\n"
            "Do you accept these terms?"
        ),
        color=discord.Color.blue()
    )
    view = discord.ui.View(timeout=60)

    async def accept_callback(i: discord.Interaction):
        data["channels"][str(interaction.channel.id)] = {
            "lang": lang,
            "guild_id": interaction.guild.id,
            "channel_id": interaction.channel.id
        }
        save_data(data)
        try:
            topic_text = f"This channel is synced via the cross-server bot.\nLanguage: {lang}\nPowered by: GloBot"
            await interaction.channel.edit(topic=topic_text)
        except discord.Forbidden:
            debug_log("error", f"Cannot edit channel topic in {interaction.channel.name}")
        await i.response.edit_message(content="Channel added to the sync system.", embed=None, view=None)
        debug_log("mod", f"Channel {interaction.channel.name} added to sync in {interaction.guild.name}")

    async def decline_callback(i: discord.Interaction):
        await i.response.edit_message(content="Action cancelled.", embed=None, view=None)

    accept_btn = discord.ui.Button(label="Accept", style=discord.ButtonStyle.success)
    decline_btn = discord.ui.Button(label="Decline", style=discord.ButtonStyle.danger)
    accept_btn.callback = accept_callback
    decline_btn.callback = decline_callback
    view.add_item(accept_btn)
    view.add_item(decline_btn)

    await interaction.followup.send(embed=embed, view=view, ephemeral=True)

@bot.tree.command(name="removechannel", description="Remove this channel from synced channels.")
async def remove_channel(interaction: discord.Interaction):
    cid = str(interaction.channel.id)
    if cid in data["channels"]:
        del data["channels"][cid]
        save_data(data)
        await interaction.response.send_message("Channel removed successfully.", ephemeral=True)
    else:
        await interaction.response.send_message("This channel isn't synced.", ephemeral=True)

@bot.tree.command(name="setlogschannel", description="Set this channel as the server's log channel.")
async def set_logs(interaction: discord.Interaction):
    data["logs"][str(interaction.guild.id)] = interaction.channel.id
    save_data(data)
    await interaction.response.send_message("Logs channel set.", ephemeral=True)

@bot.tree.command(name="warnings", description="Check a user's warning count.")
@app_commands.describe(user="User to check")
async def warnings(interaction: discord.Interaction, user: discord.Member):
    count = get_warnings(interaction.guild.id, user.id)
    embed = discord.Embed(title="User Warnings", color=discord.Color.orange())
    embed.add_field(name="User", value=user.mention)
    embed.add_field(name="Warnings", value=str(count))
    await interaction.response.send_message(embed=embed)

@bot.tree.command(name="stats", description="View current bot statistics.")
async def stats(interaction: discord.Interaction):
    await interaction.response.defer(ephemeral=True)
    active_channels = len(data["channels"])
    total_servers = len(bot.guilds)
    languages = [info["lang"] for info in data["channels"].values()]
    lang_counter = Counter(languages)
    most_used_langs = "\n".join(f"{lang}: {count}" for lang, count in lang_counter.most_common(5)) or "None"

    embed = discord.Embed(title="Bot Statistics", color=discord.Color.green())
    embed.add_field(name="Active Synced Channels", value=str(active_channels), inline=False)
    embed.add_field(name="Servers Bot is In", value=str(total_servers), inline=False)
    embed.add_field(name="Most Used Languages", value=most_used_langs, inline=False)
    await interaction.followup.send(embed=embed, ephemeral=True)

@bot.tree.command(name="help", description="Get information about the bot and how to use it.")
async def help_command(interaction: discord.Interaction):
    await interaction.response.defer(ephemeral=True)
    embed = discord.Embed(
        title="Cross-Server Sync Bot Help",
        description="Sync channels across servers, translate messages, and moderate toxic content.",
        color=discord.Color.blue()
    )
    embed.add_field(
        name="Commands",
        value=(
            "/addchannel [lang] - Add channel to sync.\n"
            "/removechannel - Remove channel from sync.\n"
            "/setlogschannel - Set logs channel.\n"
            "/warnings [user] - Check user warnings.\n"
            "/stats - Bot statistics.\n"
            "/help - This message.\n"
            "/announce [target/all] [message] - Owner announcement."
        ),
        inline=False
    )
    await interaction.followup.send(embed=embed, ephemeral=True)

@bot.tree.command(name="announce", description="Send a message to a user or all server owners (bot owner only).")
@app_commands.describe(target="User ID or 'all'", message="Message to send")
async def announce(interaction: discord.Interaction, target: str, message: str):
    if interaction.user.id != BOT_OWNER_ID:
        await interaction.response.send_message("No permission.", ephemeral=True)
        return

    message = message.replace("\\n", "\n")
    message = message.replace("\r\n", "\n").replace("\r", "\n")

    content = f"**Announcement from GloBot:**\n{message}"
    sent = 0

    if target.lower() == "all":
        for guild in bot.guilds:
            try:
                owner = guild.owner or await bot.fetch_user(guild.owner_id)
                if owner:
                    await owner.send(content)
                    sent += 1
            except discord.Forbidden:
                debug_log("error", f"Cannot DM owner of {guild.name}")

    else:
        try:
            user = await bot.fetch_user(int(target))
            await user.send(content)
            sent = 1
        except Exception as e:
            debug_log("error", f"Failed to send DM: {e}")
            await interaction.response.send_message("Could not send message.", ephemeral=True)
            return

    await interaction.followup.send(f"Sent announcement to {sent} recipient(s).", ephemeral=True)

# -------------------- Message Event --------------------
@bot.event
async def on_message(message: discord.Message):
    if message.author.bot:
        return
    guild_name = message.guild.name if message.guild else "DM"
    channel_name = message.channel.name if hasattr(message.channel, "name") else "DM"
    debug_log("all", f"Received message in {guild_name}#{channel_name} from {message.author}")

    if str(message.channel.id) not in data["channels"]:
        return

    if re.search(r"https?://", message.content):
        debug_log("mod", f"Skipped link in {message.guild.name}")
        return

    if ENABLE_TOXICITY:
        toxic_flag, scores = await is_toxic(message.content)
        if toxic_flag:
            await add_warning(message.guild.id, message.author.id)
            await log_toxic_message(message.guild, message.author, message, scores)
            await message.delete()
            return

    lang = data["channels"][str(message.channel.id)]["lang"]
    translated_text = await translate_text(message.content, lang)

    for cid, info in data["channels"].items():
        if int(cid) == message.channel.id:
            continue
        guild = bot.get_guild(info["guild_id"])
        if not guild:
            continue
        target_channel = guild.get_channel(info["channel_id"])
        if not target_channel:
            continue
        target_lang = info["lang"]
        translated_text = await translate_text(message.content, target_lang)
        await send_to_channel(target_channel, message.author, translated_text)

# -------------------- Bot Ready --------------------
@bot.event
async def on_ready():
    await bot.tree.sync()
    print(f"Logged in as {bot.user}")
    for guild in bot.guilds:
        await check_permissions(guild)

bot.run(TOKEN)
