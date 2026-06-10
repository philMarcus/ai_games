"""Competitor: a (model, thinking-config, temperature) entrant. The same weights
at different settings are DIFFERENT competitors — "gemma-think" and "gemma-nothink"
can play each other and have separate standings rows. Defined ad hoc from CLI
flags or by name in a YAML roster."""

from __future__ import annotations

import sys
from dataclasses import dataclass

import yaml

from core.ollama import resolve_model


@dataclass
class Competitor:
    label: str          # unique display name / identity (standings & transcripts use this)
    model: str          # resolved Ollama model tag
    think: bool = True
    effort: str = None  # "low"|"medium"|"high" or None (default thinking)
    temperature: float = 0.7

    @property
    def is_human(self):
        """A competitor named 'human' is played from the keyboard, not Ollama."""
        return self.model.lower() == "human"

    def think_mode(self):
        return self.effort if self.effort else ("on" if self.think else "off")

    def desc(self):
        if self.is_human:
            return "human at the keyboard"
        return " · ".join([self.model, f"think={self.think_mode()}",
                           f"t={self.temperature:g}"])

    def to_cfg(self):
        return {"model": self.model, "think": self.think_mode(),
                "temperature": self.temperature}


def parse_think_mode(mode):
    """'off' -> (False, None); 'on' -> (True, None); 'low|medium|high' -> (True, level)."""
    m = str(mode).strip().lower()
    if m in ("off", "false", "no", "none", "0"):
        return False, None
    if m in ("on", "true", "yes", "1"):
        return True, None
    if m in ("low", "medium", "high"):
        return True, m
    raise ValueError(f"unknown think mode '{mode}' (use off/on/low/medium/high)")


def competitor_from_cfg(label, cfg, installed):
    think, effort = parse_think_mode(cfg.get("think", "on"))
    model = resolve_model(str(cfg["model"]), explicit=True, installed=installed)
    return Competitor(label, model, think, effort, float(cfg.get("temperature", 0.7)))


def competitor_from_saved(label, cfg):
    """Rebuild a Competitor from saved tournament state (model already resolved)."""
    think, effort = parse_think_mode(cfg.get("think", "on"))
    return Competitor(label, cfg["model"], think, effort,
                      float(cfg.get("temperature", 0.7)))


def load_roster(path, installed):
    """Load a YAML roster of named competitors. Accepts either a top-level
    `competitors:` map or a bare label->config map. Returns an ordered dict."""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    entries = data.get("competitors", data)
    if not isinstance(entries, dict) or not entries:
        print(f"Roster '{path}' has no competitors.")
        sys.exit(1)
    roster = {}
    for label, cfg in entries.items():
        try:
            roster[str(label)] = competitor_from_cfg(str(label), cfg, installed)
        except (KeyError, ValueError) as e:
            print(f"Bad roster entry '{label}': {e}")
            sys.exit(1)
    return roster
