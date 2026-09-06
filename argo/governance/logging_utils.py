def print_stage_header(stage_no, title, detail=None):
    print("\n" + "=" * 88)
    print(f"STAGE {stage_no}: {title}")
    if detail:
        print(detail)
    print("=" * 88)

def print_step(stage_label, i, log):
    diff_score = getattr(log, "diff_score", None)
    diff_score_text = "None" if diff_score is None else f"{diff_score:.4f}"
    print(
        f"[{stage_label}] step={i:03d} action={log.action_name:8s} "
        f"old={log.old_score:.4f} new={log.new_score:.4f} "
        f"reward={log.reward:+.4f} user={log.user_score:+.2f} "
        f"processed={log.n_processed} selector_feedback={log.selector_updated} "
        f"reverted={log.reverted} proxy={getattr(log, 'used_proxy', False)} "
        f"diff_prefiltered={getattr(log, 'diff_prefiltered', False)} "
        f"diff_score={diff_score_text}"
    )

def print_stage_summary(stage_label, trainer, logs):
    print(f"\n[{stage_label}] summary: {trainer.summarize_logs(logs)}")
