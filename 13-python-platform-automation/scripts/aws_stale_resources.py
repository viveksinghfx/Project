#!/usr/bin/env python3
"""
aws_stale_resources.py
Reports untagged or idle AWS resources: EC2 instances, unattached EBS volumes,
unused Elastic IPs. Outputs JSON report and optionally posts to Slack.
"""
import argparse
import json
import sys
from datetime import datetime, timezone
import boto3


def get_untagged_instances(ec2, required_tag="Owner"):
    """Return EC2 instances missing a required tag."""
    stale = []
    paginator = ec2.get_paginator("describe_instances")
    for page in paginator.paginate(Filters=[{"Name": "instance-state-name", "Values": ["running", "stopped"]}]):
        for reservation in page["Reservations"]:
            for inst in reservation["Instances"]:
                tags = {t["Key"]: t["Value"] for t in inst.get("Tags", [])}
                if required_tag not in tags:
                    stale.append({
                        "resource_type": "ec2_instance",
                        "id": inst["InstanceId"],
                        "state": inst["State"]["Name"],
                        "launch_time": inst["LaunchTime"].isoformat(),
                        "missing_tag": required_tag,
                    })
    return stale


def get_unattached_volumes(ec2):
    """Return EBS volumes not attached to any instance."""
    stale = []
    paginator = ec2.get_paginator("describe_volumes")
    for page in paginator.paginate(Filters=[{"Name": "status", "Values": ["available"]}]):
        for vol in page["Volumes"]:
            age_days = (datetime.now(timezone.utc) - vol["CreateTime"]).days
            stale.append({
                "resource_type": "ebs_volume",
                "id": vol["VolumeId"],
                "size_gb": vol["Size"],
                "age_days": age_days,
                "create_time": vol["CreateTime"].isoformat(),
            })
    return stale


def get_unused_eips(ec2):
    """Return Elastic IPs not associated with any resource."""
    stale = []
    for addr in ec2.describe_addresses()["Addresses"]:
        if "AssociationId" not in addr:
            stale.append({
                "resource_type": "elastic_ip",
                "allocation_id": addr.get("AllocationId"),
                "public_ip": addr.get("PublicIp"),
            })
    return stale


def post_to_slack(webhook_url, report):
    """Post a summary to Slack."""
    import urllib.request
    total = sum(len(v) for v in report.values())
    text = f":warning: *Stale AWS Resources Report* — {total} resources found\n"
    for resource_type, items in report.items():
        if items:
            text += f"  • {resource_type}: {len(items)}\n"
    payload = json.dumps({"text": text}).encode()
    req = urllib.request.Request(webhook_url, data=payload, headers={"Content-Type": "application/json"})
    urllib.request.urlopen(req)


def main():
    parser = argparse.ArgumentParser(description="Report stale AWS resources")
    parser.add_argument("--region", default="ap-south-1")
    parser.add_argument("--required-tag", default="Owner")
    parser.add_argument("--slack-webhook", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", default="report.json")
    args = parser.parse_args()

    ec2 = boto3.client("ec2", region_name=args.region)

    report = {
        "untagged_instances": get_untagged_instances(ec2, args.required_tag),
        "unattached_volumes": get_unattached_volumes(ec2),
        "unused_eips": get_unused_eips(ec2),
    }

    total = sum(len(v) for v in report.values())
    print(json.dumps(report, indent=2))

    if not args.dry_run:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved to {args.output}", file=sys.stderr)

    if args.slack_webhook:
        post_to_slack(args.slack_webhook, report)
        print("Slack notification sent.", file=sys.stderr)

    sys.exit(1 if total > 0 else 0)


if __name__ == "__main__":
    main()
