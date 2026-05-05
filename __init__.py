"""Local-only demo website for the personalized music recommender.

This package is a thin wrapper around :mod:`src.personalization`. It
adds a Flask backend, a small static frontend, and an in-memory
enriched-catalog layer so the user can search for real songs instead
of typing KGRec item IDs. None of the modelling code is modified.
"""
