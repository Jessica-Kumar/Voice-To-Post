import re
from typing import Dict, Any

def calculate_safety_score(
    generated_post: str,
    context_distance: float,
    context_text: str = ""
) -> Dict[str, Any]:
    """
    Softened deterministic scoring.
    Formula: C = 0.3(AI Confidence) + 0.3(Retrieval Relevance) + 0.3(Safety Score) + 0.1(Engagement Potential)
    """
    # 1. AI CONFIDENCE (Softened Lexical Grounding)
    if not context_text or context_distance == -1.0:
        post_word_count = len(generated_post.split())
        ai_confidence = 0.8 if 15 <= post_word_count <= 100 else 0.5
    else:
        post_words = set(re.findall(r'\b\w{4,}\b', generated_post.lower()))
        db_words = set(re.findall(r'\b\w{4,}\b', context_text.lower()))

        if not db_words:
            ai_confidence = 0.6
        else:
            # Substring matching to catch "engineer" vs "engineering"
            hits = 0
            for p_word in post_words:
                if any(p_word in d_word or d_word in p_word for d_word in db_words):
                    hits += 1

            # Only require 2 hits for max score, higher base floor
            hit_rate = hits / 2.0
            ai_confidence = min(1.0, 0.5 + (hit_rate * 0.5))

    # 2. RETRIEVAL RELEVANCE (Relaxed FAISS Bounds)
    max_d = 3.0
    if context_distance == -1.0:
        retrieval_relevance = 0.6
    else:
        retrieval_relevance = max(0.0, 1.0 - (context_distance / max_d))

    # 3. SAFETY SCORE (unchanged)
    safety_score = 1.0
    forbidden_terms = ["spam", "hate", "violence", "scam", "crypto", "giveaway"]
    if any(term in generated_post.lower() for term in forbidden_terms):
        safety_score -= 0.8
    post_length = len(generated_post)
    if post_length < 20 or post_length > 2900:
        safety_score -= 0.5
    safety_score = max(0.0, safety_score)

    # 4. ENGAGEMENT POTENTIAL (unchanged)
    engagement_potential = 0.4
    hashtag_count = generated_post.count("#")
    if 1 <= hashtag_count <= 3:
        engagement_potential += 0.3
    elif hashtag_count > 4:
        engagement_potential -= 0.2

    if any(char in generated_post for char in ["!", "?", "🚀", "💡", "🔥", "🌍", "👇", "👀"]):
        engagement_potential += 0.3

    engagement_potential = max(0.0, min(1.0, engagement_potential))

    # FINAL FORMULA
    c_score = (0.3 * ai_confidence) + (0.3 * retrieval_relevance) + (0.3 * safety_score) + (0.1 * engagement_potential)

    return {
        "final_score": round(c_score, 3),
        "breakdown": {
            "ai_confidence": round(ai_confidence, 3),
            "retrieval_relevance": round(retrieval_relevance, 3),
            "safety_score": round(safety_score, 3),
            "engagement_potential": round(engagement_potential, 3)
        }
    }
