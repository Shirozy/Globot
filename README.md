# Cross-Server Sync Bot

**Globot** allows you to sync Discord channels across multiple servers, automatically translate messages, and moderate toxic content. Perfect for multilingual communities or multi-server networks.

Try our bot [here](https://discord.com/oauth2/authorize?client_id=1358240134344081438) or join our support channel: https://discord.gg/wwURWX4mFW

---

## Features

* **Cross-server message syncing**: Messages sent in one channel can be forwarded to multiple channels in other servers.
* **Automatic translation**: Messages are translated into the language set for each channel using LibreTranslate.
* **Toxicity detection**: Uses AI (`unitary/toxic-bert`) to detect toxic messages and automatically logs/deletes them.
* **Warning system**: Users receive warnings for toxic messages; server admins can track them.
* **Slash commands**: `/addchannel`, `/removechannel`, `/setlogschannel`, `/warnings`, `/stats`, `/help`.

---

## Configure environment variables

Create a `.env` file in the root directory:

```
DISCORD_TOKEN=YOUR_BOT_TOKEN
ENABLE_TOXICITY=True
ENABLE_TRANSLATION=True
LIBRETRANSLATE_URL=http://localhost:5000/translate
DEBUG_LEVEL=mod
BOT_OWNER_ID=12345..
```

* `DISCORD_TOKEN`: Your bot’s token from the Discord Developer Portal.
* `ENABLE_TOXICITY`: Toggle toxicity detection on/off (`True` or `False`).
* `ENABLE_TRANSLATION`: Toggle translation on/off (`True` or `False`).
* `LIBRETRANSLATE_URL`: URL of your self-hosted LibreTranslate instance.

---

## Translation API Setup (LibreTranslate)

1. You need a LibreTranslate instance. You can self-host with Docker:

```bash
docker run -d -p 5000:5000 libretranslate/libretranslate
```

2. The bot will send messages to this API for translation.
3. Each channel has a target language set when using `/addchannel [lang]`.
4. The bot auto-detects the source language (`source=auto`) and translates messages to the target channel’s language.

---

## Toxicity Detection

* Uses the `unitary/toxic-bert` model from HuggingFace Transformers.
* Multi-label classification for:

```
toxic, severe_toxic, obscene, threat, insult, identity_hate
```

* Messages flagged as toxic are automatically deleted, logged in a designated logs channel, and the user receives a warning via DM.
* Warnings are tracked per-server and per-user; admins can view them with `/warnings [user]`.

---

## Running the Bot

```bash
python bot.py
```

* Make sure the `.env` file is correctly set up.
* Make sure your bot has **slash command permissions** and can manage webhooks.
* The bot will automatically sync channels that are added using `/addchannel [lang]`.

---

## Commands

| Command              | Description                                                                 |
| -------------------- | --------------------------------------------------------------------------- |
| `/addchannel [lang]` | Add a channel to sync system and set its language. TOS acceptance required. |
| `/removechannel`     | Remove this channel from the sync system.                                   |
| `/setlogschannel`    | Set this channel as the log channel for toxic messages.                     |
| `/warnings [user]`   | Check a user’s warning count.                                               |
| `/stats`             | Show statistics about the bot (active channels, most used languages, etc.). |
| `/help`              | Show help and usage instructions.                                           |

---

## Notes for Server Owners

* Messages forwarded via this bot **cannot be deleted or edited**.
* It’s the responsibility of the server owner to create rules that comply with this behavior.
* Make sure the bot has permissions to **send messages, manage webhooks, and manage channels** if you want it to update channel topics automatically.

---

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.
