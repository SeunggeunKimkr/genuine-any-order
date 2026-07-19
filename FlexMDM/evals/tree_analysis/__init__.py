"""Any-order tree-traversal metrics for FlexMDM / DC-base traces.

See ``evals/metrics.md`` for the metric definitions (CBC / RUB / OBW).

The public entry points live in:
  - ``evals.tree_analysis.visualize_final_ast_mapping``: AST -> visualization
    tree, token mapping, and the pure metric computation routines (with the
    HTML rendering helpers stripped out).
  - ``evals.tree_analysis.compute_metrics``: reads ``.pt`` trace records
    produced by ``evals.humaneval_compare.generate`` and emits per-sample
    JSON + an aggregate summary.
  - ``evals.tree_analysis.cli``: the argparse-driven CLI.
"""
