import re
import discord
import json

# Regular expressions for detecting different types of content
CODE_BLOCK_REGEX = r"```(?:([\w+]+)\n)?([\s\S]*?)```"
INLINE_CODE_REGEX = r"`([^`]+)`"
URL_REGEX = r"https?://[^\s)>]+"
BULLET_LIST_REGEX = r"(?:^|\n)[\*\-\+•] .+"

def process_code_block(match):
    """Process a code block match and format it properly"""
    # Different patterns have different group structures
    if len(match.groups()) > 1:
        lang, code = match.groups()
        
        # If no language is specified, try to detect common languages
        if not lang:
            # Detect language based on common patterns
            if re.search(r"function|const|let|var|=>", code):
                lang = "javascript"
            elif re.search(r"def |class |import |from |if __name__", code):
                lang = "python"
            elif re.search(r"<html|<div|<body|<script", code):
                lang = "html"
            elif re.search(r"\{.*\:.*\}", code) and not re.search(r"[\w]+\s*\(.*\)\s*\{", code):
                # Likely JSON but not a JS/C/Java function
                try:
                    json.loads(code.strip())
                    lang = "json"
                except:
                    pass
        
        # Return formatted code block
        if lang:
            return f"```{lang}\n{code}```"
        return f"```\n{code}```"
    else:
        # Single backtick inline code
        code = match.group(1)
        return f"`{code}`"

def format_code_blocks(text):
    """
    Format code blocks with proper syntax highlighting
    """
    # Patterns to match various code block formats
    patterns = [
        r"```([a-zA-Z0-9]+)\n([\s\S]*?)```",  # ```language\ncode```
        r"```([\s\S]*?)```",                  # ```code```
        r"`([^`]+)`"                          # `code`
    ]
    
    # Process each pattern
    for pattern in patterns:
        text = re.sub(pattern, lambda m: process_code_block(m), text)
    
    return text

def detect_content_type(response):
    """
    Detect the primary content type in a response
    
    Returns:
        str: One of 'code', 'list', 'url', 'table', 'normal'
    """
    # Check for code blocks
    code_blocks = re.findall(CODE_BLOCK_REGEX, response)
    if code_blocks:
        return 'code'
    
    # Check for bulleted lists
    bullet_lists = re.findall(BULLET_LIST_REGEX, response)
    if bullet_lists and len(bullet_lists) > 2:  # At least 3 bullet points
        return 'list'
    
    # Check for URLs
    urls = re.findall(URL_REGEX, response)
    if urls and len(urls) > 2:  # Multiple URLs
        return 'url'
    
    # Check for table-like content
    if '|' in response and '-+-' in response:
        return 'table'
    
    # Default to normal text
    return 'normal'

def create_embed_for_response(response, author_name=None, avatar_url=None):
    """
    Create an appropriate embed based on the response content
    """
    content_type = detect_content_type(response)
    
    # Create different embed types based on content
    embed = discord.Embed()
    
    if author_name:
        embed.set_author(name=author_name, icon_url=avatar_url if avatar_url else discord.Embed.Empty)
    
    if content_type == 'code':
        embed.title = "Code Snippet"
        embed.color = discord.Color.blue()
        # Code is better displayed in the regular message with formatting
        return None
    
    elif content_type == 'list':
        embed.title = "Information"
        embed.description = response[:4000]  # Discord embed description limit
        embed.color = discord.Color.green()
        
    elif content_type == 'url':
        embed.title = "Resources"
        embed.description = response[:4000]
        embed.color = discord.Color.gold()
        
        # Extract and add the first URL as an embed URL if present
        urls = re.findall(URL_REGEX, response)
        if urls:
            embed.url = urls[0]
    
    elif content_type == 'table':
        embed.title = "Data Table"
        # Tables might look better in regular text with code formatting
        return None
        
    else:  # Normal text
        # For short responses, no need for an embed
        if len(response) < 200:
            return None
            
        embed.description = response[:4000]
        embed.color = discord.Color.light_grey()
    
    return embed

def find_and_format_user_mentions(message, response, bot):
    """
    Find usernames/nicknames in the response and convert them to proper Discord mentions
    Only when the user explicitly requests to mention someone
    
    Args:
        message (discord.Message): The original message for context
        response (str): The response text that might contain user references
        bot (discord.Bot): The bot instance to access guild info
        
    Returns:
        str: Response with proper Discord user mentions
    """
    if not message.guild:
        return response  # Skip if not in a guild
    
    # First, check if the user's message contains an explicit request to mention someone
    mention_request_patterns = [
        r'mention\s+([a-zA-Z0-9_]+)',
        r'@\s*([a-zA-Z0-9_]+)',
        r'tag\s+([a-zA-Z0-9_]+)',
        r'ping\s+([a-zA-Z0-9_]+)',
        r'tell\s+([a-zA-Z0-9_]+)',
        r'(please|can you|could you)\s+(mention|tag|ping|notify)\s+([a-zA-Z0-9_]+)'
    ]
    
    # Also check for numeric user IDs in the request
    id_request_patterns = [
        r'mention\s+user\s+id\s+(\d+)',
        r'mention\s+user\s+(\d{17,20})',
        r'@(\d{17,20})'
    ]
    
    user_requested_mention = False
    requested_ids = []
    requested_usernames = []
    
    # Check if the user's message contains a mention request by username
    for pattern in mention_request_patterns:
        matches = re.finditer(pattern, message.content, re.IGNORECASE)
        for match in matches:
            user_requested_mention = True
            # Extract the username from the match
            if len(match.groups()) > 2 and '(please|can you|could you)' in pattern:
                # This is for the longer pattern with 3 capture groups
                requested_usernames.append(match.group(3))
            else:
                # Standard pattern with 1 capture group
                requested_usernames.append(match.group(1))
    
    # Check if the user's message contains a mention request by user ID
    for pattern in id_request_patterns:
        matches = re.finditer(pattern, message.content, re.IGNORECASE)
        for match in matches:
            user_requested_mention = True
            requested_ids.append(match.group(1))
    
    # Only process mentions if explicitly requested
    if not user_requested_mention:
        return response
    
    # Process direct ID mentions first if any were found
    for user_id in requested_ids:
        try:
            # Try to get member by ID
            member = message.guild.get_member(int(user_id))
            if member and not member.bot:
                # Check if member can see the channel
                permissions = message.channel.permissions_for(member)
                if permissions.read_messages:
                    mention_str = f"<@{member.id}>"
                    # Look for patterns like "mention user id 123456789"
                    id_patterns = [
                        f"mention user id {user_id}",
                        f"mention user {user_id}",
                        f"@{user_id}"
                    ]
                    for pattern in id_patterns:
                        if pattern.lower() in response.lower():
                            response = re.sub(re.escape(pattern), mention_str, response, flags=re.IGNORECASE)
                    print(f"Converted user ID {user_id} to mention: {mention_str}")
        except Exception as e:
            print(f"Error processing user ID mention: {e}")
    
    # Get the current channel to check permissions
    channel = message.channel
    
    # Get all members in the guild who can see the current channel
    guild_members = []
    for member in message.guild.members:
        # Check if member can view the channel (has read message permissions)
        permissions = channel.permissions_for(member)
        if permissions.read_messages and not member.bot:
            guild_members.append(member)
    
    # Process explicitly requested usernames first
    for username in requested_usernames:
        username = username.lower().strip()
        matched_member = None
        
        # First try to find by exact username
        for member in guild_members:
            # Check for exact username match
            if member.name.lower() == username:
                matched_member = member
                break
            # Check for exact nickname match
            if member.nick and member.nick.lower() == username:
                matched_member = member
                break
            # Check for exact display_name match
            if member.display_name.lower() == username:
                matched_member = member
                break
        
        # If no exact match, try partial match
        if not matched_member:
            for member in guild_members:
                # Check if the username contains the search term
                if username in member.name.lower():
                    matched_member = member
                    break
                # Check if nickname contains the search term
                if member.nick and username in member.nick.lower():
                    matched_member = member
                    break
                # Check if display_name contains the search term
                if username in member.display_name.lower():
                    matched_member = member
                    break
        
        # If we found a matching member, look for any text patterns that might be referring to this user
        # and replace them with proper mentions
        if matched_member:
            mention_str = f"<@{matched_member.id}>"
            
            # Enhanced patterns to detect more reference styles in the bot's response
            user_reference_patterns = [
                rf'@{re.escape(username)}',                       # @username
                rf'mention\s+{re.escape(username)}',              # mention username
                rf'<@{re.escape(username)}>',                     # <@username>
                rf'<{re.escape(username)}>',                      # <username>
                rf'(?:tell|ping|tag|notify)\s+{re.escape(username)}',  # tell/ping username
                rf'(?:hey|hi|hello)\s+{re.escape(username)}',     # hey username
                rf'{re.escape(username)}(?=\s|,|\.|:|$)',         # username (as a word)
            ]
            
            # Check each pattern and replace with proper mention if found
            for pattern in user_reference_patterns:
                response = re.sub(pattern, mention_str, response, flags=re.IGNORECASE)
            
            print(f"Converted all references to '{username}' to mention: {mention_str}")
    
    # Extract all potential usernames to mention from the bot's response
    mention_patterns = [
        r'@([a-zA-Z0-9_]+)',                         # @username
        r'mention\s+(?:user\s+)?([a-zA-Z0-9_]+)',    # mention user username
        r'(?:tell|ask|ping|notify|tag)\s+([a-zA-Z0-9_]+)'  # tell/ask/ping username
    ]
    
    # Process all potential mentions in the response
    for pattern in mention_patterns:
        # Find all matches in the response
        matches = re.finditer(pattern, response, re.IGNORECASE)
        
        for match in matches:
            # Get the username from the match groups (handle different pattern group layouts)
            if len(match.groups()) > 2 and pattern.startswith('(please|can you|could you)'):
                # This is for the longer pattern with 3 capture groups
                username = match.group(3)
            else:
                # Standard pattern with 1 capture group
                username = match.group(1)
            
            username = username.lower().strip()
            matched_member = None
            
            # First try to find by exact username
            for member in guild_members:
                # Check for exact username match
                if member.name.lower() == username:
                    matched_member = member
                    break
                # Check for exact nickname match
                if member.nick and member.nick.lower() == username:
                    matched_member = member
                    break
                # Check for exact display_name match
                if member.display_name.lower() == username:
                    matched_member = member
                    break
            
            # If no exact match, try partial match
            if not matched_member:
                for member in guild_members:
                    # Check if the username contains the search term
                    if username in member.name.lower():
                        matched_member = member
                        break
                    # Check if nickname contains the search term
                    if member.nick and username in member.nick.lower():
                        matched_member = member
                        break
                    # Check if display_name contains the search term
                    if username in member.display_name.lower():
                        matched_member = member
                        break
            
            # If we found a matching member, replace the text with a proper mention
            if matched_member:
                mention_str = f"<@{matched_member.id}>"
                # Create a cleaner replacement by matching the original case
                original_match_text = match.group(0)
                response = response.replace(original_match_text, mention_str)
                print(f"Converted '{original_match_text}' to mention: {mention_str}")
    
    return response

def format_response_for_discord(response, use_embeds=False, author_name=None, avatar_url=None):
    """
    Format a response for optimal display in Discord
    
    Args:
        response (str): The text response to format
        use_embeds (bool): Whether to use Discord embeds (for rich formatting)
        author_name (str, optional): Name to show in the embed author field
        avatar_url (str, optional): URL of the avatar to use in the embed
        
    Returns:
        tuple: (content, embed) 
            - content is the text to send
            - embed is an optional Discord embed for rich formatting
    """
    # Format code blocks with proper syntax highlighting
    formatted_response = format_code_blocks(response)
    
    # Default to text response
    if not use_embeds:
        # Check if response is too long
        if len(formatted_response) > 2000:
            # If too long, use an embed instead
            embed = create_embed_for_response(formatted_response, author_name, avatar_url)
            return "", embed
        else:
            return formatted_response, None
    
    # Use embed if specifically requested
    embed = create_embed_for_response(formatted_response, author_name, avatar_url)
    return "", embed 