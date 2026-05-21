#!/usr/bin/env python3
"""
k8s_health_report.py
Scans all Kubernetes namespaces for unhealthy pods (CrashLoopBackOff, OOMKilled,
Error, Pending > threshold). Posts a structured report to Slack or stdout.
"""
import argparse
import json
import sys
from datetime import datetime, timezone
import urllib.request
from kubernetes import client, config


UNHEALTHY_REASONS = {"CrashLoopBackOff", "OOMKilled", "Error", "ImagePullBackOff", "ErrImagePull"}


def get_unhealthy_pods(v1, namespace=""):
    """Return unhealthy pods across all (or one) namespace."""
    unhealthy = []
    kwargs = {} if not namespace else {"namespace": namespace}
    pods = v1.list_pod_for_all_namespaces(**kwargs) if not namespace else v1.list_namespaced_pod(namespace)

    for pod in pods.items:
        issues = []

        # Check container statuses
        for cs in pod.status.container_statuses or []:
            state = cs.state
            if state.waiting and state.waiting.reason in UNHEALTHY_REASONS:
                issues.append({"container": cs.name, "reason": state.waiting.reason,
                                "message": state.waiting.message})
            if state.terminated and state.terminated.reason in UNHEALTHY_REASONS:
                issues.append({"container": cs.name, "reason": state.terminated.reason,
                                "exit_code": state.terminated.exit_code})
            if cs.restart_count > 5:
                issues.append({"container": cs.name, "reason": "HighRestartCount",
                                "restart_count": cs.restart_count})

        # Pending pods stuck for more than 5 minutes
        if pod.status.phase == "Pending":
            age_minutes = 0
            if pod.metadata.creation_timestamp:
                age_minutes = (datetime.now(timezone.utc) -
                               pod.metadata.creation_timestamp).seconds // 60
            if age_minutes > 5:
                issues.append({"reason": "PendingTooLong", "age_minutes": age_minutes})

        if issues:
            unhealthy.append({
                "namespace": pod.metadata.namespace,
                "pod": pod.metadata.name,
                "node": pod.spec.node_name,
                "phase": pod.status.phase,
                "issues": issues,
            })

    return unhealthy


def format_slack_message(unhealthy):
    if not unhealthy:
        return {"text": ":white_check_mark: All pods healthy across all namespaces."}

    blocks = [{"type": "header", "text": {"type": "plain_text",
               "text": f":red_circle: K8s Health Report — {len(unhealthy)} unhealthy pods"}}]

    for pod in unhealthy[:20]:  # Slack has block limits
        issue_text = "\n".join(f"  • {i['reason']}" for i in pod["issues"])
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn",
                     "text": f"*{pod['namespace']}/{pod['pod']}*\n{issue_text}"}
        })

    if len(unhealthy) > 20:
        blocks.append({"type": "section",
                        "text": {"type": "mrkdwn",
                                 "text": f"_...and {len(unhealthy) - 20} more_"}})

    return {"blocks": blocks}


def main():
    parser = argparse.ArgumentParser(description="Kubernetes pod health report")
    parser.add_argument("--namespace", default="", help="Namespace to scan (default: all)")
    parser.add_argument("--slack-webhook", default=None)
    parser.add_argument("--in-cluster", action="store_true", help="Use in-cluster kubeconfig")
    args = parser.parse_args()

    if args.in_cluster:
        config.load_incluster_config()
    else:
        config.load_kube_config()

    v1 = client.CoreV1Api()
    unhealthy = get_unhealthy_pods(v1, args.namespace)

    report = {"timestamp": datetime.now(timezone.utc).isoformat(), "unhealthy_pods": unhealthy}
    print(json.dumps(report, indent=2))

    if args.slack_webhook:
        payload = json.dumps(format_slack_message(unhealthy)).encode()
        req = urllib.request.Request(args.slack_webhook, data=payload,
                                     headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req)
        print("Slack notification sent.", file=sys.stderr)

    sys.exit(1 if unhealthy else 0)


if __name__ == "__main__":
    main()
