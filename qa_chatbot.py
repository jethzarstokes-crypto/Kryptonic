#!/usr/bin/env python3
import re, json, unicodedata
from difflib import SequenceMatcher, get_close_matches
from pathlib import Path

def normalize_q(s: str) -> str:
    s = s.lower()
    s = unicodedata.normalize("NFKD", s)
    s = s.replace("crypto currency", "cryptocurrency")
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def load_index(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def find_answer(user_q: str, qa_map: dict, cutoff: float = 0.72):
    key = normalize_q(user_q)
    if key in qa_map:
        item = qa_map[key]
        return item["answer"], item["original_question"], 1.0

    for k, item in qa_map.items():
        if key and (key in k or k in key):
            score = SequenceMatcher(None, key, k).ratio()
            if score >= cutoff * 0.9:
                return item["answer"], item["original_question"], score

    keys = list(qa_map.keys())
    candidates = get_close_matches(key, keys, n=3, cutoff=cutoff)
    if candidates:
        best = max(candidates, key=lambda x: SequenceMatcher(None, key, x).ratio())
        score = SequenceMatcher(None, key, best).ratio()
        item = qa_map[best]
        return item["answer"], item["original_question"], score

    return None, None, 0.0

def main():
    qa_map = load_index("qa_index.json")
    print("Crypto Q&A Chatbot (type 'exit' to quit)")
    while True:
        try:
            user_q = input("\nYou: ").strip()
        except EOFError:
            break
        if not user_q:
            continue
        if user_q.lower() in ("exit", "quit", "q"):
            break
        ans, matched_q, score = find_answer(user_q, qa_map)
        if ans:
            print("\nBot (matched: '{}' | confidence={:.2f}):\n{}\n".format(matched_q, score, ans))
        else:
            print("\nBot: Sorry, I couldn't find an exact answer. Try rephrasing or asking a different question.\n")

if __name__ == "__main__":
    main()
