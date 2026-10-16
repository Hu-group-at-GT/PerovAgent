from __future__ import annotations

import argparse
import json
import logging
import os
import sys

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")

from perovagent.discovery.blackboard import Blackboard, Budget
from perovagent.discovery.experiment import ExperimentRunner
from perovagent.discovery.orchestrator import DiscoveryOrchestrator
from perovagent.agents.decider import LLMDecider
from perovagent.agents.summarizer import LLMSummarizer
from perovagent.config import load_config


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/config.yml")
    p.add_argument("--output_dir", default="outputs/perovagent/discovery_run")
    p.add_argument("--n_workers", type=int, default=1,
                   help="parallel relaxation workers on a single GPU")
    p.add_argument("--gpu", type=int, default=None,
                   help="which GPU this campaign runs on, as an index into the devices. "
                   "Each run only uses one GPU, so set this for running multiple campaigns in "
                   "parallel on a multi-GPU machine.")
    p.add_argument("--random_seed", type=int, required=True,
                   help="controls the ligand conformer generation, the ligand placement during slab construction, "
                   "and partitioning of the ligand pool during retrospective test.")
    p.add_argument("--evals_cap", type=int, default=200, help="max molecules evaluated before STOP")
    p.add_argument("--iters_cap", type=int, default=150, help="max ReAct iterations before STOP")
    p.add_argument("--patience", type=int, default=0,
                   help="early-stop if no improvement after this many iterations; "
                   "0 (default) disables it")
    return p


def main() -> int:
    args = _build_parser().parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: needs OPENAI_API_KEY in env (source .env).", file=sys.stderr)
        return 2

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    cfg = load_config(args.config)
    c = cfg.campaign

    os.makedirs(args.output_dir, exist_ok=True)

    # Snapshot the config + resolved CLI run-level values
    snapshot = {"config": cfg.model_dump(),
                "cli": {"output_dir": args.output_dir, "n_workers": args.n_workers,
                        "gpu": args.gpu, "random_seed": args.random_seed,
                        "evals_cap": args.evals_cap, "iters_cap": args.iters_cap,
                        "patience": args.patience}}
    with open(os.path.join(args.output_dir, "config.json"), "w") as fh:
        json.dump(snapshot, fh, indent=2)

    print(f"[discovery] out={args.output_dir} model={c.model} "
          f"gpu={os.environ.get('CUDA_VISIBLE_DEVICES', '<all visible>')} "
          f"seed={args.random_seed} placement_seed_base={100 * args.random_seed} "
          f"budget(evals={args.evals_cap}, iters={args.iters_cap}"
          f"{f', patience={args.patience}' if args.patience > 0 else ''})", flush=True)

    experiment = ExperimentRunner(cfg, output_dir=args.output_dir, n_workers=args.n_workers,
                                  seed=args.random_seed)
    print("[discovery] " + experiment.system_context.replace("\n", "\n[discovery] "), flush=True)
    decider = LLMDecider(model=c.model, effort=c.reasoning_effort,
                         system_context=experiment.system_context,
                         material=cfg.construction.material, b_site=cfg.construction.B,
                         max_binding_energy=cfg.binding.max_binding_energy,
                         retrospective_min_per_group=c.retrospective_min_per_group,
                         prospective_min_per_group=c.prospective_min_per_group,
                         min_anchor_stratum=c.min_anchor_stratum,
                         effect_size_threshold=c.effect_size_threshold)
    summarizer = LLMSummarizer(model=c.model, effort=c.reasoning_effort,
                               effect_size_threshold=c.effect_size_threshold)

    # preflight each component
    for name, component in (("decider", decider), ("proposer", experiment.proposer),
                            ("summarizer", summarizer), ("evaluator", experiment.evaluator)):
        ok, err = component.preflight()
        if not ok:
            print(f"ERROR: {name} preflight failed: {err}", file=sys.stderr)
            return 3
        print(f"[discovery] {name} preflight OK.", flush=True)

    bb = Blackboard(goal=("Discover X-site passivant ligands (anionic OR neutral donors) whose "
                          "surface binding energy is as STRONG (as negative) as possible -- "
                          "MAXIMIZE binding strength."),
                    budget=Budget(evals_cap=args.evals_cap, iters_cap=args.iters_cap))
    orchestrator = DiscoveryOrchestrator(
        experiment, decider, blackboard=bb, log_dir=args.output_dir, summarizer=summarizer,
        patience=args.patience, effect_size_threshold=c.effect_size_threshold,
        retrospective_min_per_group=c.retrospective_min_per_group,
        prospective_min_per_group=c.prospective_min_per_group,
        min_anchor_stratum=c.min_anchor_stratum,
        significance_alpha=c.significance_alpha, random_seed=args.random_seed)
    if args.patience > 0:
        print(f"[discovery] early stopping ON: STOP after {args.patience} iterations with no "
              "improvement in the best binder", flush=True)
    try:
        orchestrator.run()
    finally:
        experiment.evaluator.close()

    s = orchestrator.summary()
    print("\n================ DISCOVERY SUMMARY ================", flush=True)
    print(f"evaluated={s['n_evaluated']}  families={s['n_families']}  strongest={s['best_metric']}", flush=True)
    print(f"principles: {s['n_principles']} proposed, {s['n_supported']} SUPPORTED", flush=True)
    print(f"stopped: {s['stop_reason']}", flush=True)
    for pr in s["supported_principles"]:
        print(f"  [{pr['validation']}] effect {pr['effect_size']:+.2f} eV "
              f"(n={pr['n_match']}/{pr['n_other']}): {pr['statement']}", flush=True)
    print("best molecules (strongest binding):", flush=True)
    for m in s["best_molecules"][:5]:
        print(f"  {m['anchor']:14s} {m['binding_energy']:+.2f} eV  {m['smiles']}", flush=True)
    print(f"\nartifacts in {args.output_dir}/ (decision_log.jsonl, principles.jsonl, summary.json)",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
