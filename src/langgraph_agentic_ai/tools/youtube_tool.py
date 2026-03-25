import re
from langchain_core.tools import tool


@tool
def get_youtube_transcript(video_url: str) -> str:
    """Get the transcript/subtitles of a YouTube video from its URL or video ID. Useful for analyzing or summarizing video content."""
    try:
        from youtube_transcript_api import YouTubeTranscriptApi

        # Extract video ID from various URL formats
        video_id = _extract_video_id(video_url)
        if not video_id:
            return f"Could not extract video ID from: {video_url}"

        ytt_api = YouTubeTranscriptApi()
        transcript = ytt_api.fetch(video_id)
        text = " ".join(snippet.text for snippet in transcript)
        # Truncate to avoid token overflow
        return text[:3000] if text else "No transcript available for this video."
    except ImportError:
        return "Error: youtube-transcript-api package not installed."
    except Exception as e:
        return f"Error fetching transcript: {str(e)[:200]}"


def _extract_video_id(url_or_id: str) -> str:
    """Extract YouTube video ID from URL or return as-is if already an ID."""
    url_or_id = url_or_id.strip()

    # Already a bare video ID (11 chars)
    if re.match(r"^[\w-]{11}$", url_or_id):
        return url_or_id

    # Standard URL patterns
    patterns = [
        r"(?:youtube\.com/watch\?v=)([\w-]{11})",
        r"(?:youtu\.be/)([\w-]{11})",
        r"(?:youtube\.com/embed/)([\w-]{11})",
        r"(?:youtube\.com/v/)([\w-]{11})",
        r"(?:youtube\.com/shorts/)([\w-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url_or_id)
        if match:
            return match.group(1)
    return ""
