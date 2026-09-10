"""Unit tests for prompts/prompt_templates.py"""

import config
from prompts.prompt_templates import HR_SYSTEM_PROMPT


def test_hr_system_prompt_is_string():
    """HR_SYSTEM_PROMPT should be a string."""
    assert isinstance(HR_SYSTEM_PROMPT, str)


def test_hr_system_prompt_is_not_empty():
    """HR_SYSTEM_PROMPT should not be empty."""
    assert len(HR_SYSTEM_PROMPT.strip()) > 0


def test_hr_system_prompt_matches_config():
    """HR_SYSTEM_PROMPT should be the same value as config.SYSTEM_PROMPT."""
    assert HR_SYSTEM_PROMPT == config.SYSTEM_PROMPT


def test_hr_system_prompt_mentions_hr():
    """HR_SYSTEM_PROMPT should contain HR-related guidance keywords."""
    prompt_lower = HR_SYSTEM_PROMPT.lower()
    assert any(keyword in prompt_lower for keyword in ["hr", "policy", "assistant"])
