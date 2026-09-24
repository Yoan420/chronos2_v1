"""Research-only comparison of raw LoRA and its genuine OOF residual stage."""

from __future__ import annotations

import argparse
import json

from chronos2_exogenous.research_oof_evaluation import evaluate_research_oof


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-directory", required=True)
    parser.add_argument("--calibration-directory", required=True)
    parser.add_argument("--calibration-panel", required=True)
    parser.add_argument("--calibration-panel-audit", required=True)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--zone", required=True)
    parser.add_argument("--reference-run", action="append", default=[], metavar="LABEL=PATH",
                        help="Reference LoRA brute facultative, appariement exact heures/origines/prix.")
    parser.add_argument("--reference-actual-policy", choices=("strict", "canonical_recompute"), default="strict",
                        help="canonical_recompute: recalcul explicite des scores des references sur la cible commune, sans modifier leurs artefacts.")
    parser.add_argument("--report-title", default="LoRA météo — correcteur OOF, recherche rolling 365 jours")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    references = {}
    for value in args.reference_run:
        label, separator, path = value.partition("=")
        if not separator or not label.strip() or not path.strip() or label in references:
            raise SystemExit("Chaque --reference-run doit etre un LABEL=PATH unique.")
        references[label] = path
    result = evaluate_research_oof(
        config_path=args.config, run_directory=args.run_directory,
        calibration_directory=args.calibration_directory,
        calibration_panel_path=args.calibration_panel,
        calibration_panel_audit_path=args.calibration_panel_audit,
        output_directory=args.output_directory, zone=args.zone,
        reference_runs=references, reference_actual_policy=args.reference_actual_policy,
        report_title=args.report_title,
    )
    print(json.dumps({"report": str(result.report_path), "manifest": str(result.manifest_path),
                      "metrics": result.metrics, "promotion_performed": False},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
