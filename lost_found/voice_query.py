import re


DEFAULT_TARGETS = ("bottle", "backpack", "cup", "person")

ITEM_ALIASES = {
    "bottle": (
        "bottle",
        "water bottle",
        "drink bottle",
        "hydro flask",
        "flask",
    ),
    "backpack": (
        "backpack",
        "back pack",
        "bag",
        "bookbag",
        "book bag",
        "school bag",
    ),
    "cup": (
        "cup",
        "mug",
        "coffee cup",
    ),
    "person": (
        "person",
        "human",
        "someone",
    ),
}

GUIDE_WORDS = ("guide", "take me", "show me", "bring me", "lead me", "go to", "find it")
CANCEL_WORDS = ("nevermind", "never mind", "cancel", "stop", "resume")


def normalize_text(text):
    return re.sub(r"\s+", " ", text.lower()).strip()


def extract_item(query, targets=DEFAULT_TARGETS):
    """Return the first known item mentioned in a natural-language query."""
    query = normalize_text(query)
    target_set = set(targets)

    for item, aliases in ITEM_ALIASES.items():
        if item not in target_set:
            continue
        for alias in aliases:
            if re.search(rf"\b{re.escape(alias)}s?\b", query):
                return item

    for item in targets:
        if re.search(rf"\b{re.escape(item)}s?\b", query):
            return item

    return None


def is_cancel_query(query):
    query = normalize_text(query)
    return any(word in query for word in CANCEL_WORDS)


def wants_guidance(query):
    query = normalize_text(query)
    return any(word in query for word in GUIDE_WORDS)


def format_location(location):
    if isinstance(location, str):
        return location
    if isinstance(location, (list, tuple)) and len(location) >= 2:
        return f"x={location[0]:.1f}, y={location[1]:.1f}"
    return "the saved location"


def format_answer(item, locations):
    """Create the robot's spoken/printed response for one item lookup."""
    if item is None:
        return "I can search for a bottle, backpack, cup, or person. Ask me about one of those."

    if not locations:
        return f"I have not seen a {item} yet."

    first_location = format_location(locations[0])
    if len(locations) == 1:
        return f"I saw one {item} near {first_location}."

    return f"I saw {len(locations)} {item}s. The first one is near {first_location}."


class VoiceIO:
    """Small voice wrapper with graceful fallback to typed input."""

    def __init__(self, enabled=True):
        self.enabled = enabled
        self.recognizer = None
        self.microphone = None
        self.tts_engine = None

        if not enabled:
            return

        try:
            import speech_recognition as sr

            self.recognizer = sr.Recognizer()
            self.microphone = sr.Microphone()
        except Exception:
            self.recognizer = None
            self.microphone = None

        try:
            import pyttsx3

            self.tts_engine = pyttsx3.init()
            self.tts_engine.setProperty("rate", 175)
        except Exception:
            self.tts_engine = None

    def listen(self, prompt="Ask me about an item: "):
        if self.recognizer and self.microphone:
            print("\n[VOICE] Listening. Ask: 'Have you seen a bottle?'")
            try:
                with self.microphone as source:
                    self.recognizer.adjust_for_ambient_noise(source, duration=0.4)
                    audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=5)
                text = self.recognizer.recognize_google(audio)
                print(f"[VOICE] Heard: {text}")
                return text
            except Exception as exc:
                print(f"[VOICE] Could not understand speech ({exc}). Switching to typing.")

        return input(prompt)

    def speak(self, message):
        print(f"\nRobot: {message}")
        if self.tts_engine:
            try:
                self.tts_engine.say(message)
                self.tts_engine.runAndWait()
            except Exception:
                pass
