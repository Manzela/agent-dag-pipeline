"""
Agent DAG Pipeline — Single Source of Truth for version tracking.

This version string is referenced by:
    - observability.py → trace metadata (set_run_context)
    - flywheel/data_flywheel.py → dataset metadata
    - flywheel/score_configs.py → per-score metadata
    - orchestrator.py → structured audit entries

Keep in sync with CHANGELOG.md at the repository root.
Format: Semantic Versioning (https://semver.org)

Changelog:
    3.0.0 — Open-source extraction from production ADK v2.6.
            Added: prompt_mutator, preference_pairs, 4-layer isolation.
            Sanitized: all proprietary references removed.
"""

__version__ = "3.0.0"
