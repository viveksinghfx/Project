#!/usr/bin/env python3
"""
ecr_image_cleanup.py
Deletes ECR images older than --days-old, always keeping the --keep most recent tags.
Run with --dry-run first to preview deletions.
"""
import argparse
import json
import sys
from datetime import datetime, timezone
import boto3


def get_images(ecr, repo, region):
    """Return all images sorted by push date (newest first)."""
    images = []
    paginator = ecr.get_paginator("describe_images")
    for page in paginator.paginate(repositoryName=repo):
        for img in page["imageDetails"]:
            images.append(img)
    images.sort(key=lambda x: x.get("imagePushedAt", datetime.min.replace(tzinfo=timezone.utc)),
                reverse=True)
    return images


def main():
    parser = argparse.ArgumentParser(description="Clean up old ECR images")
    parser.add_argument("--repo", required=True, help="ECR repository name")
    parser.add_argument("--region", default="ap-south-1")
    parser.add_argument("--keep", type=int, default=10, help="Keep N most recent images")
    parser.add_argument("--days-old", type=int, default=30, help="Also delete images older than N days")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    ecr = boto3.client("ecr", region_name=args.region)
    images = get_images(ecr, args.repo, args.region)

    to_keep = set(img["imageDigest"] for img in images[:args.keep])
    now = datetime.now(timezone.utc)
    to_delete = []

    for img in images:
        if img["imageDigest"] in to_keep:
            continue
        pushed = img.get("imagePushedAt")
        if pushed and (now - pushed).days >= args.days_old:
            to_delete.append({
                "imageDigest": img["imageDigest"],
                "imageTags": img.get("imageTags", []),
                "pushedAt": pushed.isoformat(),
                "age_days": (now - pushed).days,
            })

    print(f"Images to delete ({len(to_delete)}):")
    print(json.dumps(to_delete, indent=2))

    if not to_delete:
        print("Nothing to delete.")
        sys.exit(0)

    if args.dry_run:
        print("\n[DRY RUN] No images deleted.")
        sys.exit(0)

    batch = [{"imageDigest": img["imageDigest"]} for img in to_delete]
    response = ecr.batch_delete_image(repositoryName=args.repo, imageIds=batch)
    deleted = len(response.get("imageIds", []))
    failures = response.get("failures", [])
    print(f"\nDeleted: {deleted}, Failures: {len(failures)}")
    if failures:
        print(json.dumps(failures, indent=2))
        sys.exit(1)


if __name__ == "__main__":
    main()
