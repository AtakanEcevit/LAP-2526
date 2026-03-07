#!/usr/bin/env python3
"""
Biometric Few-Shot Verification — CLI Demo

Quick verification tool using trained models, no API server needed.

Usage:
    # Compare two images directly
    python demo.py compare --modality signature --model siamese \
        --image1 "data/raw/signatures/CEDAR/full_org/original_1_1.png" \
        --image2 "data/raw/signatures/CEDAR/full_org/original_1_2.png"

    # Enroll a user (store reference embeddings)
    python demo.py enroll --user "john" --modality signature --model siamese \
        --images "data/raw/signatures/CEDAR/full_org/original_1_1.png" \
                 "data/raw/signatures/CEDAR/full_org/original_1_2.png" \
                 "data/raw/signatures/CEDAR/full_org/original_1_3.png"

    # Verify a query image against an enrolled user
    python demo.py verify --user "john" \
        --image "data/raw/signatures/CEDAR/full_org/original_1_4.png"

    # List enrolled users
    python demo.py list

    # Delete an enrolled user
    python demo.py delete --user "john"
"""

import argparse
import sys
import os

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from inference.engine import VerificationEngine
from inference.enrollment_store import EnrollmentStore
from inference.config import VALID_MODALITIES, VALID_MODEL_TYPES


def cmd_compare(args):
    """Compare two images directly."""
    engine = VerificationEngine()
    engine.load(args.modality, args.model)

    print(f"\n  Comparing images:")
    print(f"    Image 1: {os.path.basename(args.image1)}")
    print(f"    Image 2: {os.path.basename(args.image2)}")
    print(f"    Model:   {args.model}/{args.modality}")
    print()

    result = engine.compare(args.image1, args.image2)

    _print_result(result)


def cmd_enroll(args):
    """Enroll a user with reference images."""
    engine = VerificationEngine()
    engine.load(args.modality, args.model)
    store = EnrollmentStore()

    print(f"\n  Enrolling user '{args.user}' ({args.modality}/{args.model})")
    print(f"  Reference images: {len(args.images)}\n")

    for img_path in args.images:
        if not os.path.exists(img_path):
            print(f"  [ERROR] File not found: {img_path}")
            sys.exit(1)

        embedding = engine.extract_embedding(img_path)
        result = store.enroll(args.user, args.modality, args.model, embedding)
        print(f"    [OK] Enrolled {os.path.basename(img_path)} "
              f"(total samples: {result['sample_count']})")

    print(f"\n  [DONE] User '{args.user}' enrolled with "
          f"{result['sample_count']} sample(s).")


def cmd_verify(args):
    """Verify a query image against an enrolled user."""
    store = EnrollmentStore()

    # Look up the user to get their modality and model_type
    user_info = store.get_user(args.user)
    if user_info is None:
        print(f"\n  [ERROR] User '{args.user}' is not enrolled.")
        print("  Run 'python demo.py list' to see enrolled users.")
        sys.exit(1)

    modality = user_info["modality"]
    model_type = user_info["model_type"]

    engine = VerificationEngine()
    engine.load(modality, model_type)

    enrolled_embeddings = store.get_embeddings(args.user)

    print(f"\n  Verifying query against user '{args.user}'")
    print(f"    Modality:    {modality}")
    print(f"    Model:       {model_type}")
    print(f"    Enrolled:    {len(enrolled_embeddings)} sample(s)")
    print(f"    Query image: {os.path.basename(args.image)}")
    print()

    result = engine.verify_against_prototype(args.image, enrolled_embeddings)

    _print_result(result, user_id=args.user)


def cmd_list(args):
    """List all enrolled users."""
    store = EnrollmentStore()
    users = store.list_users()

    if not users:
        print("\n  No users enrolled yet.")
        print("  Use 'python demo.py enroll ...' to enroll a user.\n")
        return

    print(f"\n  {'User ID':<20s} {'Modality':<14s} {'Model':<14s} "
          f"{'Samples':<10s} {'Enrolled At'}")
    print(f"  {'-'*20} {'-'*14} {'-'*14} {'-'*10} {'-'*20}")

    for u in users:
        enrolled = u.get("enrolled_at", "unknown")[:19]  # trim microseconds
        print(f"  {u['user_id']:<20s} {u['modality']:<14s} "
              f"{u['model_type']:<14s} {u['sample_count']:<10d} {enrolled}")
    print()


def cmd_delete(args):
    """Delete an enrolled user."""
    store = EnrollmentStore()
    if store.delete_user(args.user):
        print(f"\n  [DONE] User '{args.user}' deleted.\n")
    else:
        print(f"\n  [INFO] User '{args.user}' was not enrolled.\n")


def _print_result(result: dict, user_id: str = None):
    """Pretty-print a verification result."""
    # Show validation warnings if present
    validation = result.get("validation")
    if validation:
        warnings = []
        if isinstance(validation, dict):
            for key, val in validation.items():
                if isinstance(val, dict):
                    warnings.extend(val.get("warnings", []))
                elif isinstance(val, list):
                    warnings.extend(val)
        if warnings:
            print(f"  \u26a0  Validation Warnings:")
            for w in warnings:
                print(f"     \u2022 {w}")
            print()

    match_icon = ">> MATCH" if result["match"] else ">> NO MATCH"
    score_pct = int(result["score"] * 30)
    score_bar = "#" * score_pct + "." * (30 - score_pct)

    print(f"  +{'-'*50}+")
    if user_id:
        print(f"  |  User:      {user_id:<37s}|")
    print(f"  |  Result:    {match_icon:<37s}|")
    print(f"  |  Score:     {result['score']:.6f}                       |")
    print(f"  |  Threshold: {result['threshold']:.6f}                       |")
    print(f"  |  [{score_bar}] |")
    print(f"  +{'-'*50}+")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Biometric Few-Shot Verification — CLI Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo.py compare --modality signature --model siamese --image1 img1.png --image2 img2.png
  python demo.py enroll  --user john --modality face --model prototypical --images f1.jpg f2.jpg f3.jpg
  python demo.py verify  --user john --image query.jpg
  python demo.py list
  python demo.py delete  --user john
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    subparsers.required = True

    # ── compare ──────────────────────────────────────────────────────────
    p_compare = subparsers.add_parser(
        "compare", help="Compare two images directly"
    )
    p_compare.add_argument(
        "--modality", required=True,
        choices=sorted(VALID_MODALITIES),
        help="Biometric modality"
    )
    p_compare.add_argument(
        "--model", required=True,
        choices=sorted(VALID_MODEL_TYPES),
        help="Model architecture"
    )
    p_compare.add_argument("--image1", required=True, help="First image path")
    p_compare.add_argument("--image2", required=True, help="Second image path")
    p_compare.set_defaults(func=cmd_compare)

    # ── enroll ───────────────────────────────────────────────────────────
    p_enroll = subparsers.add_parser(
        "enroll", help="Enroll a user with reference images"
    )
    p_enroll.add_argument("--user", required=True, help="User ID to enroll")
    p_enroll.add_argument(
        "--modality", required=True,
        choices=sorted(VALID_MODALITIES),
        help="Biometric modality"
    )
    p_enroll.add_argument(
        "--model", required=True,
        choices=sorted(VALID_MODEL_TYPES),
        help="Model architecture"
    )
    p_enroll.add_argument(
        "--images", nargs="+", required=True,
        help="Path(s) to reference image(s)"
    )
    p_enroll.set_defaults(func=cmd_enroll)

    # ── verify ───────────────────────────────────────────────────────────
    p_verify = subparsers.add_parser(
        "verify", help="Verify a query image against an enrolled user"
    )
    p_verify.add_argument("--user", required=True, help="Enrolled user ID")
    p_verify.add_argument("--image", required=True, help="Query image path")
    p_verify.set_defaults(func=cmd_verify)

    # ── list ─────────────────────────────────────────────────────────────
    p_list = subparsers.add_parser("list", help="List enrolled users")
    p_list.set_defaults(func=cmd_list)

    # ── delete ───────────────────────────────────────────────────────────
    p_delete = subparsers.add_parser("delete", help="Delete an enrolled user")
    p_delete.add_argument("--user", required=True, help="User ID to delete")
    p_delete.set_defaults(func=cmd_delete)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
