"""
prepare_data.py

Converts the Northern Thai Dialect dataset into JSONL training examples.
Supports 4 task types trained together in one multitask run:

  1. translation_ntd_to_std  — NTD sentence -> Standard Thai (comprehension)
  2. translation_std_to_ntd  — Standard Thai -> NTD (production)
  3. response_single         — Post -> single NTD reply
  4. response_thread         — Post + previous comments -> next NTD reply

Data source:
  Sheet: 'natural'
  CI rows (Context-Independent): used for translation tasks
  CD rows (Context-Dependent):   used for response tasks (grouped by Head_Post_ID)

  Key columns:
    Text_Northern       — the NTD sentence / comment
    Text_Standard_Thai  — the Standard Thai translation (gold standard)
    Head_Post_Text      — the original post text (for CD rows)
    Head_Post_ID        — groups comments under the same post
    Contextual_Dependency — CI or CD
    Head_Post_Lang      — NTD or STD (language of the original post)

Training mix (anchored balancing):
  ~40% translation (both directions combined)
  ~30% response_single
  ~30% response_thread

Usage:
    python scripts/prepare_data.py --task multitask --input data/Master_Dataset.xlsx
    python scripts/prepare_data.py --task translation --input data/Master_Dataset.xlsx
"""

import pandas as pd
import json
import argparse
import random
import os

# ── System prompts ────────────────────────────────────────────────────────────
# Task tags help the model distinguish between tasks during multitask training.
# Clear, explicit tags prevent task confusion (e.g. translating instead of replying).

SYSTEM_NTD_TO_STD = (
    "[TASK: TRANSLATION_NTD_TO_STD]\n"
    "Translate Northern Thai (คำเมือง) into natural Standard Thai.\n"
    "Output only the translation."
)

SYSTEM_STD_TO_NTD = (
    "[TASK: TRANSLATION_STD_TO_NTD]\n"
    "Translate Standard Thai into natural Northern Thai dialect (คำเมือง).\n"
    "Output only the translation."
)

SYSTEM_RESPONSE = (
    "[TASK: RESPONSE_GENERATION]\n"
    "You are a Northern Thai social media user.\n"
    "Write a natural reply in Northern Thai dialect (คำเมือง).\n"
    "Output only the reply."
)

SYSTEM_INTENT = (
    "[TASK: INTENT_CLASSIFICATION]\n"
    "Classify the communicative intent of the given Northern Thai sentence.\n"
    "Choose from: complaint, joke, invitation, sarcasm, question, "
    "agreement, advice, information.\n"
    "Output only the intent label."
)

# ── Task builders ─────────────────────────────────────────────────────────────

def make_translation_ntd_to_std(ntd: str, std: str) -> dict:
    """
    Task 1: NTD -> Standard Thai (comprehension)
    Source: Text_Northern
    Gold:   Text_Standard_Thai
    """
    return {
        "messages": [
            {"role": "system",    "content": SYSTEM_NTD_TO_STD},
            {"role": "user",      "content": f"ข้อความ:\n{ntd}"},
            {"role": "assistant", "content": std},
        ],
        "task_type": "translation_ntd_to_std",
    }

def make_translation_std_to_ntd(ntd: str, std: str) -> dict:
    """
    Task 2: Standard Thai -> NTD (production)
    Source: Text_Standard_Thai
    Gold:   Text_Northern
    """
    return {
        "messages": [
            {"role": "system",    "content": SYSTEM_STD_TO_NTD},
            {"role": "user",      "content": f"ข้อความ:\n{std}"},
            {"role": "assistant", "content": ntd},
        ],
        "task_type": "translation_std_to_ntd",
    }

def make_response_single(post_text: str, reply_ntd: str) -> dict:
    """
    Task 3: Single-turn response generation
    Input:  one post (NTD or STD — both occur naturally in dataset)
    Output: a natural NTD reply
    Source: Head_Post_Text (post), Text_Northern (reply)
    """
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_RESPONSE},
            {
                "role": "user",
                "content": f"โพสต์:\n{post_text}\n\nตอบกลับ:"
            },
            {"role": "assistant", "content": reply_ntd},
        ],
        "task_type": "response_single",
    }

def make_response_thread(post_text: str, prior_comments: list,
                         target_reply: str) -> dict:
    """
    Task 4: Multi-turn thread response generation
    Input:  post + list of prior comments shown as a bullet list
    Output: the next NTD reply

    Design: prior comments are presented as a flat list — NOT as alternating
    user/assistant turns. This reflects the real structure of Facebook threads
    where multiple users comment, not a back-and-forth dialogue.
    No fake alternation, no dummy turns.

    Source: Head_Post_Text + Text_Northern (all comments in thread)
    Max prior comments: controlled by --max_prior (default 3)
    """
    context_block = "\n".join([f"- {c}" for c in prior_comments])

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_RESPONSE},
            {
                "role": "user",
                "content": (
                    f"โพสต์:\n{post_text}\n\n"
                    f"ความคิดเห็นก่อนหน้า:\n{context_block}\n\n"
                    f"ตอบกลับ:"
                )
            },
            {"role": "assistant", "content": target_reply},
        ],
        "task_type": "response_thread",
    }

def make_intent_example(ntd: str, intent: str) -> dict:
    """
    Intent classification (added when Intent column is populated)
    Source: Text_Northern
    Gold:   Intent
    """
    return {
        "messages": [
            {"role": "system",    "content": SYSTEM_INTENT},
            {"role": "user",      "content": f"ข้อความ:\n{ntd}"},
            {"role": "assistant", "content": intent},
        ],
        "task_type": "intent_classification",
    }

# ── Data quality check ────────────────────────────────────────────────────────

def is_valid_translation(ntd: str, std: str) -> bool:
    """
    Returns True only if this row has a genuine translation.
    Rejects rows where Gold == NTD (untranslated — Gold column copied from NTD).
    """
    ntd = ntd.strip()
    std = std.strip()
    if not ntd or not std:
        return False
    if ntd == "nan" or std == "nan":
        return False
    if ntd == std:
        return False
    return True

# ── Example builders ──────────────────────────────────────────────────────────

def build_translation_examples(df: pd.DataFrame) -> list:
    """
    Build translation examples from all rows (CI + CD).
    Only includes rows where Gold != NTD (genuine translations).
    Each valid row produces 2 examples: NTD->STD and STD->NTD.
    """
    examples = []
    skipped_untranslated = 0
    skipped_missing      = 0

    for _, row in df.iterrows():
        ntd = str(row.get("Text_Northern", "")).strip()
        std = str(row.get("Text_Standard_Thai", "")).strip()

        if not ntd or not std or ntd == "nan" or std == "nan":
            skipped_missing += 1
            continue
        if ntd == std:
            skipped_untranslated += 1
            continue

        examples.append(make_translation_ntd_to_std(ntd, std))
        examples.append(make_translation_std_to_ntd(ntd, std))

    print(f"  Translation:       {len(examples):>5} examples "
          f"({skipped_untranslated} untranslated skipped, "
          f"{skipped_missing} missing skipped)")
    return examples

def build_response_single_examples(df: pd.DataFrame) -> list:
    """
    Build single-turn response examples from CD rows.
    Post (Head_Post_Text) -> Comment (Text_Northern).
    Posts can be NTD or STD — model learns to reply in NTD either way.
    """
    examples = []
    cd = df[df["Contextual_Dependency"] == "CD"].copy()

    for _, row in cd.iterrows():
        post  = str(row.get("Head_Post_Text", "")).strip()
        reply = str(row.get("Text_Northern",  "")).strip()

        if not post or not reply or post == "nan" or reply == "nan":
            continue

        examples.append(make_response_single(post, reply))

    print(f"  Response single:   {len(examples):>5} examples")
    return examples

def build_response_thread_examples(df: pd.DataFrame,
                                   max_prior: int = 3) -> list:
    """
    Build multi-turn thread examples from CD rows grouped by Head_Post_ID.
    Uses a sliding window: for each comment in a thread, use up to max_prior
    previous comments as context and predict the next comment.

    Window examples from a 5-comment thread (max_prior=3):
      [c1] -> c2
      [c1, c2] -> c3
      [c1, c2, c3] -> c4
      [c2, c3, c4] -> c5

    Prior comments shown as a flat bullet list (reflects real Facebook structure).
    No fake user/assistant alternation. No dummy turns.
    """
    examples = []
    cd = df[df["Contextual_Dependency"] == "CD"].copy()

    for post_id, group in cd.groupby("Head_Post_ID"):
        post_text = str(group.iloc[0].get("Head_Post_Text", "")).strip()
        if not post_text or post_text == "nan":
            continue

        comments = []
        for _, row in group.iterrows():
            c = str(row.get("Text_Northern", "")).strip()
            if c and c != "nan":
                comments.append(c)

        if len(comments) < 2:
            continue

        for i in range(1, len(comments)):
            target      = comments[i]
            prior_start = max(0, i - max_prior)
            prior       = comments[prior_start:i]
            examples.append(make_response_thread(post_text, prior, target))

    print(f"  Response thread:   {len(examples):>5} examples "
          f"(from {cd['Head_Post_ID'].nunique()} threads)")
    return examples

def build_intent_examples(df: pd.DataFrame) -> list:
    """
    Build intent classification examples where Intent column is annotated.
    """
    examples = []
    intent_rows = df[df["Intent"].notna()].copy()

    for _, row in intent_rows.iterrows():
        ntd    = str(row.get("Text_Northern", "")).strip()
        intent = str(row.get("Intent", "")).strip()

        if not ntd or ntd == "nan" or not intent or intent == "nan":
            continue

        examples.append(make_intent_example(ntd, intent))

    print(f"  Intent:            {len(examples):>5} examples")
    return examples

# ── Balancing ─────────────────────────────────────────────────────────────────

def balance_examples(
    translation:     list,
    response_single: list,
    response_thread: list,
    intent:          list,
) -> list:
    """
    Balanced sampling using smallest-task anchoring.
    Anchors to the smallest of the three core tasks so no task dominates.
    Approximate target: 40% translation, 30% single, 30% thread.
    Intent examples are always kept in full (usually small).
    """
    min_size = min(
        len(translation),
        len(response_single),
        len(response_thread),
    )

    if min_size == 0:
        print("WARNING: one task has zero samples — skipping balancing")
        return translation + response_single + response_thread + intent

    n_trans  = int(min_size * 1.33)  # ~40%
    n_single = int(min_size * 1.0)   # ~30%
    n_thread = int(min_size * 1.0)   # ~30%

    sampled = (
        random.sample(translation,     min(n_trans,  len(translation)))  +
        random.sample(response_single, min(n_single, len(response_single))) +
        random.sample(response_thread, min(n_thread, len(response_thread))) +
        intent
    )

    print(f"\n  Balanced dataset (anchored to smallest task = {min_size}):")
    print(f"    Translation:       {min(n_trans,  len(translation))}")
    print(f"    Response single:   {min(n_single, len(response_single))}")
    print(f"    Response thread:   {min(n_thread, len(response_thread))}")
    print(f"    Intent:            {len(intent)}")
    print(f"    Total:             {len(sampled)}")

    return sampled

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Prepare training data for Northern Thai LLM fine-tuning"
    )
    parser.add_argument("--input",      default="data/Master_Dataset.xlsx")
    parser.add_argument("--sheet",      default="natural")
    parser.add_argument("--task",
                        choices=["translation", "response_single",
                                 "response_thread", "multitask"],
                        default="multitask")
    parser.add_argument("--split",      type=float, default=0.8,
                        help="Train proportion (remainder split evenly into valid/test)")
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--output_dir", default="data")
    parser.add_argument("--max_prior",  type=int,   default=3,
                        help="Max prior comments in thread examples (keep <=3 for 512 token limit)")
    parser.add_argument("--no_balance", action="store_true",
                        help="Skip task balancing — use all examples as-is")
    args = parser.parse_args()

    random.seed(args.seed)

    print(f"Loading {args.input} (sheet: {args.sheet})...")
    df = pd.read_excel(args.input, sheet_name=args.sheet)
    print(f"  Loaded {len(df)} rows")
    print(f"  CI rows: {len(df[df['Contextual_Dependency']=='CI'])}")
    print(f"  CD rows: {len(df[df['Contextual_Dependency']=='CD'])}")

    # ── Build examples ─────────────────────────────────────────────────────────
    print(f"\nBuilding examples (task={args.task})...")

    translation_ex     = []
    response_single_ex = []
    response_thread_ex = []
    intent_ex          = []

    if args.task in ("translation", "multitask"):
        translation_ex = build_translation_examples(df)

    if args.task in ("response_single", "multitask"):
        response_single_ex = build_response_single_examples(df)

    if args.task in ("response_thread", "multitask"):
        response_thread_ex = build_response_thread_examples(df, args.max_prior)

    if args.task == "multitask":
        intent_ex = build_intent_examples(df)

    # ── Combine ────────────────────────────────────────────────────────────────
    if args.task == "multitask" and not args.no_balance:
        all_examples = balance_examples(
            translation_ex, response_single_ex,
            response_thread_ex, intent_ex,
        )
    else:
        all_examples = (
            translation_ex + response_single_ex +
            response_thread_ex + intent_ex
        )

    # ── Shuffle and split ──────────────────────────────────────────────────────
    random.shuffle(all_examples)
    n_total = len(all_examples)
    n_train = int(n_total * args.split)
    n_valid = int(n_total * ((1 - args.split) / 2))

    train_data = all_examples[:n_train]
    valid_data = all_examples[n_train:n_train + n_valid]
    test_data  = all_examples[n_train + n_valid:]

    print(f"\n  Split: Train {len(train_data)} | "
          f"Valid {len(valid_data)} | Test {len(test_data)}")

    # ── Write JSONL files ──────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)

    def write_jsonl(data, path):
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"  Wrote {len(data):4d} examples -> {path}")

    write_jsonl(train_data, os.path.join(args.output_dir, "train.jsonl"))
    write_jsonl(valid_data, os.path.join(args.output_dir, "valid.jsonl"))
    write_jsonl(test_data,  os.path.join(args.output_dir, "test.jsonl"))

    # Task-specific test files for targeted evaluation
    test_by_task = {}
    for ex in test_data:
        t = ex.get("task_type", "unknown")
        test_by_task.setdefault(t, []).append(ex)

    print("\n  Task-specific test files:")
    for task_type, examples in sorted(test_by_task.items()):
        path = os.path.join(args.output_dir, f"test_{task_type}.jsonl")
        write_jsonl(examples, path)

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n── Dataset summary ──────────────────────────────────────────")
    task_counts = {}
    for ex in all_examples:
        t = ex.get("task_type", "unknown")
        task_counts[t] = task_counts.get(t, 0) + 1
    for t, n in sorted(task_counts.items()):
        print(f"  {t:<38} {n:>5}")
    print(f"  {'TOTAL':<38} {n_total:>5}")
    print(f"\nDone.")
    print(f"  train.jsonl / valid.jsonl   -> fine-tuning")
    print(f"  test_<task>.jsonl           -> task-specific evaluation")

if __name__ == "__main__":
    main()
