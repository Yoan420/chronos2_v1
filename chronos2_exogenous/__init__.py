"""Point-in-time exogenous inputs and governed Chronos-2 LoRA runtime.

Training, evaluation and shadow artifacts remain isolated from the business
pipeline.  Operational runners can only load the narrow production modules
after an explicit, checksum-pinned promotion and activation contract.
"""
