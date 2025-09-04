import re
from typing import Iterable


URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
EMAIL_PATTERN = re.compile(r"\b[\w.-]+@[\w.-]+\.[A-Za-z]{2,}\b")
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
NON_WORD_PATTERN = re.compile(r"[^a-z\s]")
MULTI_SPACE_PATTERN = re.compile(r"\s{2,}")


def clean_text(text: str) -> str:

    if text is None:
        return ""

    text = str(text)
    text = text.lower()
    text = URL_PATTERN.sub(" ", text)
    text = EMAIL_PATTERN.sub(" ", text)
    text = HTML_TAG_PATTERN.sub(" ", text)
    text = text.replace("\n", " ").replace("\r", " ")
    text = NON_WORD_PATTERN.sub(" ", text)
    text = MULTI_SPACE_PATTERN.sub(" ", text)
    return text.strip()


def clean_texts(texts: Iterable[str]) -> list[str]:

    return [clean_text(t) for t in texts]


