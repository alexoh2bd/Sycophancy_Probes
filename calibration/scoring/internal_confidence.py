"""
Compute internal confidence via mean log-probability of generated tokens (Rivera et al.).

Used when generating model responses; for assertion data with ground-truth labels,
this module is optional (we use CSV assertiveness directly for probe training).
"""

# Stub for future use when combining generated + scored pipeline
