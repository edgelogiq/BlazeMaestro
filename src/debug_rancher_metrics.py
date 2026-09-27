#!/usr/bin/env python3
"""
Debug script to check what metrics are available in Rancher Monitoring.

Usage:
    python debug_rancher_metrics.py [prometheus_url] [namespace]

Example:
    python debug_rancher_metrics.py http://prometheus:9090 default
"""

import sys
import os
import json
import urllib.parse
import urllib.request
from typing import List, Dict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'EdgeLogiQ', 'BlazeAIoT', 'BlazeMaestro', 'ConfigUpdater'))


def query_prometheus(prometheus_url: str, query: str) -> List[Dict]:
    """Query Prometheus HTTP API directly to avoid BlazeAIoT dependency issues."""
    params = urllib.parse.urlencode({'query': query})
    url = prometheus_url.rstrip('/') + '/api/v1/query?' + params
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            payload = response.read().decode('utf-8')
            data = json.loads(payload)
    except Exception as exc:
        raise RuntimeError(f"Prometheus query failed: {exc}") from exc

    if data.get('status') != 'success':
        raise RuntimeError(f"Prometheus query error: {data.get('error', 'unknown error')}")

    return data.get('data', {}).get('result', [])

def discover_metrics(prometheus_url: str, patterns=None):
    """Query Prometheus for available metrics matching patterns."""
    if not patterns:
        patterns = [
            'rancher_pod',
            'container_cpu',
            'container_memory', 
            'pod_cpu',
            'pod_memory',
            'kubelet',
        ]
    
    print(f"\n=== Discovering metrics in Prometheus ===\n")
    
    all_results = {}
    for pattern in patterns:
        query = f'{{__name__=~"{pattern}.*"}}'
        print(f"Querying pattern '{pattern}'...")
        try:
            results = query_prometheus(prometheus_url, query)
            if results:
                metric_names = set()
                for result in results:
                    if '__name__' in result.get('metric', {}):
                        metric_names.add(result['metric']['__name__'])
                
                all_results[pattern] = sorted(list(metric_names))
                print(f"  ✓ Found {len(metric_names)} metrics")
                
                # Sample labels for first few metrics
                sample_labels = {}
                for result in results[:3]:
                    metric_name = result.get('metric', {}).get('__name__')
                    if metric_name:
                        sample_labels[metric_name] = dict(result.get('metric', {}))
                
                if sample_labels:
                    print(f"  Sample metrics with labels:")
                    for name, labels in list(sample_labels.items())[:3]:
                        label_str = ', '.join(f'{k}={v}' for k, v in list(labels.items())[:5])
                        print(f"    - {name}: {label_str}")
            else:
                print(f"  ✗ No metrics found")
                all_results[pattern] = []
        except Exception as e:
            print(f"  ✗ Error: {e}")
            all_results[pattern] = []
    
    return all_results

def main():
    prometheus_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:9090"
    namespace = sys.argv[2] if len(sys.argv) > 2 else "default"
    
    print(f"Prometheus URL: {prometheus_url}")
    print(f"Target namespace: {namespace}")
    
    # Discover metrics
    results = discover_metrics(prometheus_url)
    
    # Check specific Rancher metrics
    print(f"\n=== Checking specific Rancher pod metrics ===\n")
    
    rancher_queries = [
        ('rancher_pod_cpu_usage', f'rancher_pod_cpu_usage{{namespace="{namespace}"}}', 'Rancher pod CPU usage'),
        ('rancher_pod_memory_usage_bytes', f'rancher_pod_memory_usage_bytes{{namespace="{namespace}"}}', 'Rancher pod memory usage'),
        ('container_cpu_usage_seconds_total', f'container_cpu_usage_seconds_total{{namespace="{namespace}"}}', 'Container CPU usage (total)'),
        ('container_memory_working_set_bytes', f'container_memory_working_set_bytes{{namespace="{namespace}"}}', 'Container memory working set'),
    ]
    
    for metric_name, query, description in rancher_queries:
        print(f"\n{description} ({metric_name}):")
        print(f"  Query: {query}")
        try:
            results = query_prometheus(prometheus_url, query)
            if results:
                print(f"  ✓ Found {len(results)} results")
                # Show first result sample
                if results:
                    first = results[0]
                    labels_str = ', '.join(f'{k}={v}' for k, v in first.get('metric', {}).items())
                    value = first['value'][1] if first.get('value') else 'N/A'
                    print(f"    Sample: [{labels_str}] = {value}")
            else:
                print(f"  ✗ No results")
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    # Suggestions
    print(f"\n=== Troubleshooting suggestions ===\n")
    print("If you see 0 results for Rancher metrics:")
    print("1. Check if Rancher Monitoring is actually installed: kubectl get -n cattle-monitoring-system pods")
    print("2. Verify Prometheus can scrape the metrics: check targets at /prometheus/targets")
    print("3. Check pod labels in your namespace: kubectl get pods -n {namespace} --show-labels")
    print("4. Try querying just '{{__name__=~\".*cpu.*\"}}' to see all CPU-related metrics")
    print("5. Check Rancher Monitoring ServiceMonitor objects: kubectl get servicemonitor -A")

if __name__ == '__main__':
    main()
