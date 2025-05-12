import re

def filter_thinking(response):
    """
    Filter out thinking tags from the response.
    Removes anything between <think> and </think> tags.
    """
    # Pattern to match <think>...</think> including the tags and all content inside
    pattern = r'<think>.*?</think>'
    # Replace with empty string (using re.DOTALL to match across multiple lines)
    filtered_response = re.sub(pattern, '', response, flags=re.DOTALL)
    return filtered_response.strip()

def split_response(response, max_length=1999):
    lines = response.splitlines()
    chunks = []
    current_chunk = ""

    for line in lines:
        if len(current_chunk) + len(line) + 1 > max_length:
            chunks.append(current_chunk.strip())
            current_chunk = line
        else:
            if current_chunk:
                current_chunk += "\n"
            current_chunk += line

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks