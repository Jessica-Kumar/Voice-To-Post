"""
Content Moderation Service - FREE
Filters profanity, hate speech, and inappropriate content from generated posts.
Uses multiple free methods for comprehensive moderation.
"""

import logging
import re
from typing import Dict, Any, List, Tuple
from better_profanity import profanity

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize profanity filter
profanity.load_censor_words()

# Additional hate speech patterns (basic, extend as needed)
HATE_SPEECH_PATTERNS = [
    # Slurs and discriminatory language (partial list for demonstration)
    r'\b(racist|sexist|homophobic|transphobic)\b',
    r'\b(hate|kill|attack)\s+(all\s+)?(women|men|blacks|whites|jews|muslims|christians|gays|trans)',
    # Violence incitement
    r'\b(attack|hurt|kill|harm|destroy)\s+(them|those|people)',
    # Extreme political rhetoric
    r'\b(genocide|exterminate|cleanse)\b'
]

# Suspicious financial/scam patterns
SCAM_PATTERNS = [
    r'\b(send\s+money|wire\s+transfer|cash\s+app|venmo\s+me)\b',
    r'\b(guaranteed\s+profit|get\s+rich\s+quick|100%\s+returns?)\b',
    r'\b(dm\s+for\s+proof|click\s+link\s+in\s+bio)\b',
    r'\b(crypto\s+giveaway|free\s+bitcoin|double\s+your\s+money)\b'
]

# Self-harm/danger patterns
SELF_HARM_PATTERNS = [
    r'\b(kill\s+myself|end\s+my\s+life|commit\s+suicide)\b',
    r'\b(harm\s+myself|hurt\s+myself)\b'
]


def moderate_content(text: str, platform: str = "twitter") -> Dict[str, Any]:
    """
    Moderate content for inappropriate material.

    Args:
        text: The text to moderate
        platform: Target platform (for platform-specific rules)

    Returns:
        {
            "is_safe": bool,
            "issues_found": List[str],
            "severity": str,  # "safe", "warning", "blocked"
            "cleaned_text": str,  # Text with profanity censored
            "recommendations": List[str]
        }
    """
    issues = []
    severity = "safe"
    recommendations = []

    # 1. Check for profanity
    if profanity.contains_profanity(text):
        issues.append("profanity")
        severity = "warning"
        recommendations.append("Contains profanity - consider rephrasing for wider audience")

    # 2. Check for hate speech patterns
    text_lower = text.lower()
    for pattern in HATE_SPEECH_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            issues.append("hate_speech")
            severity = "blocked"
            recommendations.append("Contains potential hate speech - content blocked")
            break

    # 3. Check for scam patterns
    for pattern in SCAM_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            issues.append("potential_scam")
            severity = "warning"
            recommendations.append("Contains patterns common in scams - review carefully")
            break

    # 4. Check for self-harm content
    for pattern in SELF_HARM_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            issues.append("self_harm")
            severity = "blocked"
            recommendations.append("Contains self-harm content - content blocked")
            break

    # 5. Platform-specific checks
    if platform == "linkedin":
        # LinkedIn is more professional
        casual_issues = check_professionalism(text)
        if casual_issues:
            issues.extend(casual_issues)
            if severity == "safe":
                severity = "warning"
            recommendations.append("Consider more professional tone for LinkedIn")

    # 6. Check for excessive caps (spam indicator)
    caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    if caps_ratio > 0.5 and len(text) > 20:
        issues.append("excessive_caps")
        if severity == "safe":
            severity = "warning"
        recommendations.append("Excessive capitalization detected - may appear as spam")

    # 7. Check for excessive special characters (spam indicator)
    special_chars = len(re.findall(r'[!@#$%^&*()_+=\[\]{};:"|<>?]', text))
    if special_chars > len(text) * 0.2:
        issues.append("excessive_symbols")
        if severity == "safe":
            severity = "warning"
        recommendations.append("Excessive special characters - may trigger spam filters")

    # Generate cleaned text (profanity censored)
    cleaned_text = profanity.censor(text)

    return {
        "is_safe": severity != "blocked",
        "issues_found": issues,
        "severity": severity,
        "cleaned_text": cleaned_text,
        "recommendations": recommendations,
        "original_length": len(text),
        "cleaned_length": len(cleaned_text)
    }


def check_professionalism(text: str) -> List[str]:
    """Check for unprofessional content on LinkedIn."""
    issues = []
    text_lower = text.lower()

    # Overly casual phrases
    casual_phrases = [
        r'\blol\b', r'\blmao\b', r'\byolo\b',
        r'\bhella\b', r'\baf\b', r'\btbh\b',
        r'\bbruh\b', r'\bfam\b'
    ]

    for phrase in casual_phrases:
        if re.search(phrase, text_lower):
            issues.append("too_casual")
            break

    return issues


def batch_moderate(texts: List[str], platform: str = "twitter") -> Dict[str, Any]:
    """
    Moderate multiple texts at once.

    Returns:
        {
            "total": int,
            "safe": int,
            "warnings": int,
            "blocked": int,
            "results": List[Dict]
        }
    """
    results = []
    safe_count = 0
    warning_count = 0
    blocked_count = 0

    for text in texts:
        result = moderate_content(text, platform)
        results.append(result)

        if result["severity"] == "safe":
            safe_count += 1
        elif result["severity"] == "warning":
            warning_count += 1
        else:
            blocked_count += 1

    return {
        "total": len(texts),
        "safe": safe_count,
        "warnings": warning_count,
        "blocked": blocked_count,
        "results": results
    }


def get_moderation_summary(moderation_result: Dict[str, Any]) -> str:
    """Generate human-readable summary of moderation results."""
    if moderation_result["is_safe"]:
        if moderation_result["severity"] == "safe":
            return "✅ Content is safe to post"
        else:
            issues = ", ".join(moderation_result["issues_found"])
            return f"⚠️ Content has minor issues: {issues}. Review recommended."
    else:
        issues = ", ".join(moderation_result["issues_found"])
        return f"🚫 Content blocked due to: {issues}"


def apply_content_policy(text: str, policy: str = "strict") -> Dict[str, Any]:
    """
    Apply content policy to text.

    Args:
        text: Text to check
        policy: "strict" (block warnings too) or "permissive" (only block severe)

    Returns:
        {
            "approved": bool,
            "moderation": Dict,
            "final_text": str
        }
    """
    moderation = moderate_content(text)

    if policy == "strict":
        # Block both warnings and blocked content
        approved = moderation["severity"] == "safe"
        final_text = moderation["cleaned_text"] if approved else ""
    else:
        # Only block severe issues
        approved = moderation["severity"] != "blocked"
        final_text = moderation["cleaned_text"] if approved else ""

    return {
        "approved": approved,
        "moderation": moderation,
        "final_text": final_text
    }


# Initialize profanity filter on import
logger.info("✅ Content moderation service initialized with better-profanity")
