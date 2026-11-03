"""Cron prompt threat scanning: the small user-authored prompt gets the strict pattern set;
the assembled prompt (with skill bodies) gets only prose-proof directives."""

import logging
import re

# Single source of truth shared with the install-time scanner (skills_guard): a narrower
# cron-local copy once let obfuscated directives slip past this runtime tripwire.
from tools.threat_patterns import INVISIBLE_CHARS as _CRON_INVISIBLE_CHARS

# Logger parity with the origin module (these functions used to log there).
logger = logging.getLogger("tools.cronjob_tools")

# Strict patterns — user prompt only. A directive-shaped cron prompt has no business
# containing `cat ~/.hermes/.env` or `rm -rf /`; there it is a smoking gun, not prose.
# Two threat surfaces, two scanners: 1. `_scan_cron_prompt()` runs against this at create/update time and as
# a runtime defense-in-depth. 2. Assembled prompt that includes loaded skill content (large markdown bodies,
# often security docs, postmortems, runbooks discussing attack patterns in PROSE). Reusing the strict
# patterns here false-positives every time a skill *describes* a command — see #3968 follow-up: the
# `hermes-agent-dev` skill contains a security postmortem mentioning `cat ~/.hermes/.env`, which tripped
# `read_secrets` and silently killed all PR-scout jobs. Skill bodies are user-curated and scanned at install
# time by `skills_guard.py`. The runtime cron scan only needs to catch the patterns whose phrasing does NOT
# survive normal English prose: classic prompt-injection directives ("ignore previous instructions",
# "disregard your rules"), deception directives, and invisible unicode. `_scan_cron_skill_assembled()` runs
# against the assembled prompt with this tighter pattern set. Both scanners share the invisible-unicode
# check and the GitHub Authorization header exemption.
_CRON_THREAT_PATTERNS = [
    (r'ignore\s+(?:\w+\s+)*(?:previous|all|above|prior)\s+(?:\w+\s+)*instructions', "prompt_injection"),
    (r'do\s+not\s+tell\s+the\s+user', "deception_hide"),
    (r'system\s+prompt\s+override', "sys_prompt_override"),
    (r'disregard\s+(your|all|any)\s+(instructions|rules|guidelines)', "disregard_rules"),
    (r'cat\s+[^\n]*(\.env|credentials|\.netrc|\.pgpass|id_rsa|id_ed25519|id_ecdsa)', "read_secrets"),
    (r'authorized_keys', "ssh_backdoor"), (r'/etc/sudoers|visudo', "sudoers_mod"),
    (r'rm\s+-rf\s+/', "destructive_root_rm"),
]

# Looser set for the assembled prompt: command-shape patterns are dropped because skill
# markdown (postmortems, runbooks) legitimately *describes* those commands and skill bodies
# are vetted at install time — only unambiguous injection directives remain.
_CRON_SKILL_ASSEMBLED_PATTERNS = _CRON_THREAT_PATTERNS[:4]

# Quote marks (straight + curly + backtick) recognised on either side of a matched phrase —
# mirrors tools/skills_tool_plugin.py's _is_quoted_citation (t_be9d36f3).
_QUOTE_CHARS = frozenset('"\'\u201c\u201d\u2018\u2019`')
# Punctuation allowed between the end of a matched phrase and its closing quote mark, e.g.
# the trailing comma in `"ignore previous instructions,"` — a citation, not a live directive.
_TRAILING_PUNCT = frozenset(',.;:!?)]} \t')

# Pattern IDs where a quoted citation (both sides quote-bracketed on the same line) is exempt
# from the block: these are canonical injection-PHRASE directives, the exact false-positive
# class t_be9d36f3 already fixed for the skill security scanner — a postmortem, a false-positive
# writeup, or (recursively, t_6aa22272) this very fix's own kanban history quoting the phrase as
# a documented example can never execute as a live directive. Command-shape / exfiltration
# patterns (_CRON_EXFIL_COMMAND_PATTERNS, and read_secrets/ssh_backdoor/sudoers_mod/
# destructive_root_rm below) stay unconditional — a quoted curl/cat/rm command is the same bytes
# if it were ever executed, so quoting buys nothing there. Never widen this set to loosen real
# detection (same principle prepper-disinfo-defence documents).
_QUOTE_AWARE_IDS = frozenset({"prompt_injection", "deception_hide", "sys_prompt_override", "disregard_rules"})


def _is_quoted_citation(line: str, start: int, end: int) -> bool:
    """True when ``line[start:end]`` (a threat-pattern match) is bracketed by quote marks on
    the same line — a documented example/citation, not an injected directive."""
    before = line[:start].rstrip()
    if not before or before[-1] not in _QUOTE_CHARS:
        return False
    after = line[end:]
    i = 0
    while i < len(after) and after[i] in _TRAILING_PUNCT:
        i += 1
    return i < len(after) and after[i] in _QUOTE_CHARS


_CRON_SECRET_VAR_RE = r'\$\{?\w*(?:KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|API)\w*\}?'
# Obvious leak paths only: secret in the destination URL, in a POST/form body, or in an
# Authorization header to an arbitrary host.
_CRON_EXFIL_COMMAND_PATTERNS = [
    (rf'curl\s+[^\n]*https?://[^\s"\'`]*{_CRON_SECRET_VAR_RE}', "exfil_curl_url"),
    (rf'wget\s+[^\n]*https?://[^\s"\'`]*{_CRON_SECRET_VAR_RE}', "exfil_wget_url"),
    (rf'curl\s+[^\n]*(?:--data(?:-raw|-binary|-urlencode)?|-d|--form|-F)\s+[^\n]*{_CRON_SECRET_VAR_RE}', "exfil_curl_data"),
    (rf'wget\s+[^\n]*--post-(?:data|file)=[^\n]*{_CRON_SECRET_VAR_RE}', "exfil_wget_post"),
    (rf'curl\s+[^\n]*(?:-H|--header)\s+["\']Authorization:\s*(?:Bearer|token)\s+{_CRON_SECRET_VAR_RE}["\']', "exfil_curl_auth_header"),
]

# U+200D (ZWJ) is a required part of many emoji sequences (👨‍👩‍👧, 🏳️‍🌈): block it
# between plain text, allow it inside an emoji grapheme cluster.
_EMOJI_NEIGHBOUR_CP_RANGES = ((0x1F000, 0x1FFFF), (0x2600, 0x27BF), (0x2300, 0x23FF), (0x1F1E6, 0x1F1FF), (0x20E3, 0x20E3))
_VARIATION_SELECTOR_CP = 0xFE0F


def _zwj_has_emoji_neighbour(text: str, idx: int) -> bool:
    """True when the ZWJ at text[idx] sits between emoji codepoints (skipping VS16)."""
    left = idx - 1
    while left >= 0 and ord(text[left]) == _VARIATION_SELECTOR_CP:
        left -= 1
    right = idx + 1
    while right < len(text) and ord(text[right]) == _VARIATION_SELECTOR_CP:
        right += 1
    if left < 0 or right >= len(text):
        return False
    return all(
        any(lo <= ord(text[pos]) <= hi for lo, hi in _EMOJI_NEIGHBOUR_CP_RANGES) for pos in (left, right)
    )


def _strip_cron_safe_constructs(prompt: str) -> str:
    """Scrub the bundled GitHub skill's `Authorization: token $GITHUB_TOKEN` + api.github.com
    curl so it doesn't trip the auth-header exfil rule.

    re.sub scrubs EVERY occurrence. The trailing ``[^\\s;&|$`]*`` consumes only the URL path —
    never separators or subshell openers — so a payload smuggled onto the same line still gets
    scanned. Host must be exactly api.github.com followed by ``/``, whitespace, quote, or end:
    lookalike authorities (api.github.com.evil.com, api.github.com@evil.com) fall through.
    """
    return re.sub(
        rf'curl\s+[^\n;&|$`]*(?:-H|--header)\s+["\']Authorization:\s*token\s+{_CRON_SECRET_VAR_RE}["\']'
        r'\s+["\']?https://api\.github\.com(?::\d+)?(?:/|\s|$|["\'])[^\s;&|$`]*',
        'curl https://api.github.com/user',
        prompt,
        flags=re.IGNORECASE,
    )


def _strip_invisible_unicode(prompt: str) -> tuple[str, list[str]]:
    """Strip invisible-unicode chars, keeping ZWJ inside legitimate emoji.

    Returns ``(cleaned, sorted U+XXXX labels removed)``. The skills-attached path sanitizes
    (a stray zero-width space in vetted skill content must not permanently kill the job).
    """
    if not prompt:
        return prompt, []
    removed: set[str] = set()
    cleaned: list[str] = []
    for idx, ch in enumerate(prompt):
        if ch in _CRON_INVISIBLE_CHARS and not (ch == '\u200d' and _zwj_has_emoji_neighbour(prompt, idx)):
            removed.add(f"U+{ord(ch):04X}")
            continue
        cleaned.append(ch)
    return ''.join(cleaned), sorted(removed)


def _first_pattern_error(text: str, *pattern_sets) -> str:
    for patterns in pattern_sets:
        for pattern, pid in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                if pid in _QUOTE_AWARE_IDS:
                    line_start = text.rfind('\n', 0, match.start()) + 1
                    line_end = text.find('\n', match.end())
                    if line_end == -1:
                        line_end = len(text)
                    line = text[line_start:line_end]
                    if _is_quoted_citation(line, match.start() - line_start, match.end() - line_start):
                        continue  # cited example, not a live directive — keep scanning
                return (
                    f"Blocked: prompt matches threat pattern '{pid}'. Cron prompts must not "
                    "contain injection or exfiltration payloads."
                )
    return ""


def _scan_cron_prompt(prompt: str) -> str:
    """Strict scan of the USER-SUPPLIED prompt (create/update + runtime defense-in-depth).
    Returns an error string when blocked, else "". Invisible unicode is reported first, in
    ``_CRON_INVISIBLE_CHARS`` order (emoji ZWJ allowed)."""
    prompt_to_scan = _strip_cron_safe_constructs(prompt)
    removed = set(_strip_invisible_unicode(prompt_to_scan)[1])
    for char in _CRON_INVISIBLE_CHARS:
        if f"U+{ord(char):04X}" in removed:
            return f"Blocked: prompt contains invisible unicode U+{ord(char):04X} (possible injection)."
    return _first_pattern_error(prompt_to_scan, _CRON_THREAT_PATTERNS, _CRON_EXFIL_COMMAND_PATTERNS)


def _scan_cron_skill_assembled(assembled: str) -> tuple[str, str]:
    """Loose scan of the ASSEMBLED prompt (skill content included). Invisible unicode is
    SANITIZED (stripped + logged), not blocked — the hard block stays on raw user prompts,
    the actual injection surface. Returns ``(cleaned_prompt, error)``, error "" when passed."""
    cleaned, removed = _strip_invisible_unicode(assembled)
    if removed:
        logger.warning(
            "Cron skill-assembled prompt: stripped %d invisible-unicode "
            "char(s) (%s) from vetted skill content",
            len(removed), ", ".join(removed),
        )
    return cleaned, _first_pattern_error(_strip_cron_safe_constructs(cleaned), _CRON_SKILL_ASSEMBLED_PATTERNS)
