"""Synthetic dataset generator.

Pipeline per company: plan -> true tree (structure) -> canonical titles (titles)
-> people attributes (people) -> missing-employee simulation and observed-view
noise (noise) -> rows (company). `generator.generate_dataset` orchestrates and
writes observed and truth tables to separate directories.
"""
