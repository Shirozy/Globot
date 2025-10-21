import discord
from discord import app_commands
from discord.ext import commands
import asyncio, aiohttp, json, os
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from collections import Counter
import torch

# Load environment variables
load_dotenv()
TOKEN = os.getenv("DISCORD_TOKEN")
ENABLE_TOXICITY = os.getenv("ENABLE_TOXICITY", "True").lower() == "true"
ENABLE_TRANSLATION = os.getenv("ENABLE_TRANSLATION", "True").lower() == "true"
TRANSLATE_URL = os.getenv("LIBRETRANSLATE_URL", "http://localhost:5000/translate")

INTENTS = discord.Intents.default()
INTENTS.message_content = True
INTENTS.guilds = True
INTENTS.messages = True

bot = commands.Bot(command_prefix="!", intents=INTENTS)

# -------------------- Persistent storage --------------------
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

# -------------------- Helper functions --------------------

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
        except:
            return text
    return text

async def is_toxic(text: str) -> bool:
    if not ENABLE_TOXICITY:
        return False
    scores = await asyncio.to_thread(classify_text, text)
    return any(v > 0.5 for v in scores.values()), scores

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

async def add_warning(guild_id: int, user_id: int):
    warnings = data["warnings"].setdefault(str(guild_id), {})
    warnings[str(user_id)] = warnings.get(str(user_id), 0) + 1
    save_data(data)

def get_warnings(guild_id: int, user_id: int) -> int:
    return data["warnings"].get(str(guild_id), {}).get(str(user_id), 0)

# -------------------- Webhook sending --------------------

async def send_to_channel(channel: discord.TextChannel, author: discord.User, content: str):
    try:
        # Check if a webhook is cached
        wh_id = data["webhooks"].get(str(channel.id))
        webhook = None
        if wh_id:
            try:
                webhook = await bot.fetch_webhook(wh_id)
            except discord.NotFound:
                webhook = None

        # If no webhook exists, create a new one
        if not webhook:
            webhook = await channel.create_webhook(name="SyncBot")
            data["webhooks"][str(channel.id)] = webhook.id
            save_data(data)

        # Send message
        await webhook.send(
            content=content,
            username=author.display_name[:32],
            avatar_url=author.display_avatar.url,
            allowed_mentions=discord.AllowedMentions.none()
        )
    except discord.Forbidden:
        print(f"⚠️ Cannot send webhook message in {channel.name}")
    except Exception as e:
        print(f"⚠️ Webhook error in {channel.name}: {e}")

# -------------------- Slash commands --------------------

@bot.tree.command(name="addchannel", description="Add this channel to the cross-server sync and set its language.")
@app_commands.describe(lang="Language code (e.g., en, es, fr)")
async def add_channel(interaction: discord.Interaction, lang: str):
    await interaction.response.defer(ephemeral=True)

    embed = discord.Embed(
        title="Terms of Service for Cross-Server Sync Bot",
        description=(
            "By using this bot in your channel, you agree to the following terms:\n\n"
            "**Content Restrictions:**\n"
            "- NSFW, hateful, or illegal content is strictly prohibited.\n"
            "- Toxic messages may be automatically deleted.\n\n"
            "**Message Handling:**\n"
            "- Messages sent via this bot may be forwarded to other servers and channels.\n"
            "- Once a message is sent through this bot, it cannot be deleted or edited.\n"
            "- It is the responsibility of the server owner to define rules for channels using this bot and enforce compliance.\n\n"
            "**Moderation:**\n"
            "- Toxic messages will trigger warnings and may be logged.\n"
            "- Repeated violations may result in removal from the sync system.\n\n"
            "Do you accept these terms?"
        ),
        color=discord.Color.blue()
    )

    view = discord.ui.View(timeout=60)

    async def accept_callback(i: discord.Interaction):
        # Store the channel info
        data["channels"][str(interaction.channel.id)] = {
            "lang": lang,
            "guild_id": interaction.guild.id,
            "channel_id": interaction.channel.id
        }
        save_data(data)

        # Update channel topic
        try:
            topic_text = f"This channel is synced via the cross-server bot.\nLanguage: {lang}\nPowered by: GloBot"
            await interaction.channel.edit(topic=topic_text)
        except discord.Forbidden:
            pass  # Skip if bot cannot edit channel

        await i.response.edit_message(
            content="Channel has been successfully added to the sync system.",
            embed=None,
            view=None
        )

    async def decline_callback(i: discord.Interaction):
        await i.response.edit_message(
            content="Action cancelled. Channel was not added.",
            embed=None,
            view=None
        )

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
        await interaction.response.send_message("The channel has been removed successfully.", ephemeral=True)
    else:
        await interaction.response.send_message("This channel isn't synced.", ephemeral=True)

@bot.tree.command(name="setlogschannel", description="Set this channel as the server's log channel.")
async def set_logs(interaction: discord.Interaction):
    data["logs"][str(interaction.guild.id)] = interaction.channel.id
    save_data(data)
    await interaction.response.send_message("This channel is now set as the logs channel.", ephemeral=True)

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

    # Active channels
    active_channels = len(data["channels"])

    # Number of servers the bot is in
    total_servers = len(bot.guilds)

    # Most used languages
    languages = [info["lang"] for info in data["channels"].values()]
    lang_counter = Counter(languages)
    most_used_langs = "\n".join(f"{lang}: {count}" for lang, count in lang_counter.most_common(5)) or "None"

    embed = discord.Embed(
        title="Bot Statistics",
        color=discord.Color.green()
    )
    embed.add_field(name="Active Synced Channels", value=str(active_channels), inline=False)
    embed.add_field(name="Servers Bot is In", value=str(total_servers), inline=False)
    embed.add_field(name="Most Used Languages", value=most_used_langs, inline=False)

    # You can add more fields in the future here:
    # embed.add_field(name="Most Active Servers", value="Coming Soon", inline=False)
    # embed.add_field(name="Total Messages Forwarded", value="Coming Soon", inline=False)

    await interaction.followup.send(embed=embed, ephemeral=True)

@bot.tree.command(name="help", description="Get information about the bot and how to use it.")
async def help_command(interaction: discord.Interaction):
    await interaction.response.defer(ephemeral=True)

    embed = discord.Embed(
        title="Cross-Server Sync Bot Help",
        description="This bot allows you to sync channels across multiple servers, automatically translate messages, and moderate toxic content.",
        color=discord.Color.blue()
    )

    # Commands Overview
    embed.add_field(
        name="Commands",
        value=(
            "/addchannel [lang] - Add this channel to the sync system and set the language for translations. You will be prompted to accept the Terms of Service.\n"
            "/removechannel - Remove this channel from the sync system.\n"
            "/setlogschannel - Set this channel as the log channel for toxic messages.\n"
            "/warnings [user] - Check the warning count of a user.\n"
            "/stats - View statistics about the bot usage.\n"
            "/help - Show this help message."
        ),
        inline=False
    )

    # Features
    embed.add_field(
        name="Features",
        value=(
            "• Cross-server channel syncing via webhooks.\n"
            "• Automatic message translation using LibreTranslate.\n"
            "• Toxicity detection and moderation using Toxic-BERT.\n"
            "• Logging toxic messages to a specified channel.\n"
            "• Warning system for users who send toxic messages.\n"
            "• Automatic webhook creation and management.\n"
            "• Professional Terms of Service enforcement."
        ),
        inline=False
    )

    # Usage Notes
    embed.add_field(
        name="Usage Notes",
        value=(
            "• Messages sent via this bot may be forwarded to other servers and channels.\n"
            "• Once a message is sent through this bot, it cannot be deleted or edited.\n"
            "• Server owners are responsible for creating rules that comply with these limitations.\n"
            "• Only users with the appropriate permissions can add or remove channels.\n"
            "• The bot respects channel languages set via `/addchannel`."
        ),
        inline=False
    )

    # Footer
    embed.set_footer(text="For support or questions, contact the bot developer.")

    await interaction.followup.send(embed=embed, ephemeral=True)

# -------------------- Message event --------------------

@bot.event
async def on_message(message: discord.Message):
    if message.author.bot:
        return

    if str(message.channel.id) not in data["channels"]:
        return

    # ---------------- Toxicity check ----------------
    if ENABLE_TOXICITY:
        toxic_flag, scores = await is_toxic(message.content)
        if toxic_flag:
            await add_warning(message.guild.id, message.author.id)
            await log_toxic_message(message.guild, message.author, message, scores)
            await message.delete()

            count = get_warnings(message.guild.id, message.author.id)
            embed = discord.Embed(title="Toxic Message Removed", color=discord.Color.red())
            embed.add_field(name="Your Message", value=message.content or "[No content]")
            embed.add_field(name="Warning Count", value=str(count))
            try:
                await message.author.send(embed=embed)
            except discord.Forbidden:
                log_channel_id = data["logs"].get(str(message.guild.id))
                if log_channel_id:
                    log_channel = message.guild.get_channel(log_channel_id)
                    if log_channel:
                        await log_channel.send(
                            f"Could not DM {message.author.mention} about a deleted toxic message."
                        )
            return  # Don't forward toxic messages

    # ---------------- Translation ----------------
    lang = data["channels"][str(message.channel.id)]["lang"]
    text = await translate_text(message.content, lang) if ENABLE_TRANSLATION else message.content

    # ---------------- Forward to other servers ----------------
    for cid, info in data["channels"].items():
        # skip same channel
        if int(cid) == message.channel.id:
            continue
        guild = bot.get_guild(info["guild_id"])
        if not guild:
            continue
        target_channel = guild.get_channel(info["channel_id"])
        if not target_channel:
            continue

        # translate into the target channel's language
        target_lang = info["lang"]
        translated_text = await translate_text(message.content, target_lang) if ENABLE_TRANSLATION else message.content

        await send_to_channel(target_channel, message.author, translated_text)


# -------------------- Bot ready --------------------

@bot.event
async def on_ready():
    await bot.tree.sync()
    print(f"Logged in as {bot.user}")

bot.run(TOKEN)
