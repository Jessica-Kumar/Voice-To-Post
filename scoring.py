import re
from typing import Dict, Any


def calculate_safety_score(
    generated_post: str,
    context_distance: float,
    context_text: str = ""
) -> Dict[str, Any]:
    """
    Per-post scoring that produces meaningfully different scores across variations.

    Formula: 0.3*ai_confidence + 0.3*retrieval_relevance + 0.3*safety_score + 0.1*engagement_potential
    """

    post_words = set(re.findall(r'\b\w{4,}\b', generated_post.lower()))

    # ── 1. AI CONFIDENCE (post length + structure quality) ──────────────────
    # Measures how "complete" and well-formed this specific post is.
    post_length = len(generated_post)

    if post_length < 20:
        length_score = 0.3
    elif post_length < 60:
        length_score = 0.6
    elif post_length <= 280:
        # Twitter-range: good
        length_score = 0.9
    elif post_length <= 1500:
        # LinkedIn-range: ideal
        length_score = 1.0
    elif post_length <= 2900:
        # Acceptable but long
        length_score = 0.8
    else:
        length_score = 0.3

    # Sentence variety: more sentences = more structured post
    sentence_count = len(re.findall(r'[.!?]+', generated_post))
    structure_bonus = min(0.1, sentence_count * 0.02)

    ai_confidence = min(1.0, length_score + structure_bonus)

    # ── 2. RETRIEVAL RELEVANCE (how much the post actually uses context) ─────
    if not context_text or context_distance == -1.0:
        # No context available — give a neutral mid score, not a free pass.
        retrieval_relevance = 0.70
    else:
        db_words = set(re.findall(r'\b\w{4,}\b', context_text.lower()))

        if not db_words or not post_words:
            retrieval_relevance = 0.65
        else:
            # FIX: divide by post_words count (was wrongly dividing by 2.0)
            hits = sum(
                1 for p_word in post_words
                if any(p_word in d_word or d_word in p_word for d_word in db_words)
            )
            hit_rate = hits / len(post_words)  # 0.0 – 1.0

            # Distance factor: closer retrieval = more relevant
            max_d = 3.0
            distance_factor = max(0.0, 1.0 - (context_distance / max_d))

            # Combine: 60% overlap, 40% distance
            retrieval_relevance = min(1.0, (hit_rate * 0.6) + (distance_factor * 0.4))

    # ── 3. SAFETY SCORE ──────────────────────────────────────────────────────
    safety_score = 1.0
    forbidden_terms = ["spam", "hate", "violence", "scam", "crypto", "giveaway"]
    if any(term in generated_post.lower() for term in forbidden_terms):
        safety_score -= 0.8
    if post_length < 20 or post_length > 2900:
        safety_score -= 0.5
    safety_score = max(0.0, safety_score)

    # ── 4. ENGAGEMENT POTENTIAL (per-post — varies based on actual content) ──
    engagement_potential = 0.3  # lower base so there's more room to differentiate

    hashtag_count = generated_post.count("#")
    if 1 <= hashtag_count <= 3:
        engagement_potential += 0.25
    elif hashtag_count == 4:
        engagement_potential += 0.10
    elif hashtag_count > 4:
        engagement_potential -= 0.15  # hashtag stuffing

    # Emojis
    emoji_matches = re.findall(r'[🚀💡🔥🌍👇👀✅🎯📌💬🤔⚡🙌]', generated_post)
    emoji_count = len(emoji_matches)
    if 1 <= emoji_count <= 3:
        engagement_potential += 0.20
    elif emoji_count > 4:
        engagement_potential += 0.05  # too many = spammy

    # Punctuation hooks: questions or exclamations
    if '?' in generated_post:
        engagement_potential += 0.10
    if '!' in generated_post:
        engagement_potential += 0.05

    # Call to action words
    cta_words = ['comment', 'share', 'follow', 'tag', 'dm', 'link', 'read', 'watch', 'join', 'try']
    if any(w in generated_post.lower() for w in cta_words):
        engagement_potential += 0.10

    engagement_potential = max(0.0, min(1.0, engagement_potential))

    # ── FINAL SCORE ──────────────────────────────────────────────────────────
    final_score = (
        0.3 * ai_confidence +
        0.3 * retrieval_relevance +
        0.3 * safety_score +
        0.1 * engagement_potential
    )

    return {
        "final_score": round(final_score, 3),
        "breakdown": {
            "ai_confidence": round(ai_confidence, 3),
            "retrieval_relevance": round(retrieval_relevance, 3),
            "safety_score": round(safety_score, 3),
            "engagement_potential": round(engagement_potential, 3),
        }
    }
