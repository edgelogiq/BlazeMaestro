#!/usr/bin/env python3
"""
Validate and inspect services_update.xml metrics structure.

Usage:
    python inspect_xml_metrics.py [path_to_services_update.xml]

Shows:
- Instances with metrics collected
- Instances without metrics (still at 0.0)
- Last update timestamps
- Suggested action items
"""

import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime

def inspect_metrics_xml(xml_path):
    """Inspect metrics in XML and report status."""
    xml_path = Path(xml_path)
    
    if not xml_path.exists():
        print(f"❌ File not found: {xml_path}")
        return
    
    print(f"\n=== XML Metrics Inspection ===\n")
    print(f"File: {xml_path}")
    print(f"Size: {xml_path.stat().st_size} bytes")
    print(f"Modified: {datetime.fromtimestamp(xml_path.stat().st_mtime)}")
    
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception as e:
        print(f"❌ Error parsing XML: {e}")
        return
    
    # Find all instances
    instances = root.findall('.//instance')
    print(f"\nTotal instances: {len(instances)}\n")
    
    # Categorize by metrics status
    with_metrics = []
    without_metrics = []
    partial_metrics = []
    
    for inst in instances:
        inst_id = inst.find('id').text if inst.find('id') is not None else 'unknown'
        node = inst.find('node').text if inst.find('node') is not None else 'unknown'
        
        metrics = {
            'exec_time': float(inst.find('exec_time').text or 0) if inst.find('exec_time') is not None else 0,
            'cpu_cores': float(inst.find('cpu_cores').text or 0) if inst.find('cpu_cores') is not None else 0,
            'ram_gb': float(inst.find('ram_gb').text or 0) if inst.find('ram_gb') is not None else 0,
            'execution_rate_hz': float(inst.find('execution_rate_hz').text or 0) if inst.find('execution_rate_hz') is not None else 0,
        }
        
        # Count non-zero metrics
        non_zero_count = sum(1 for v in metrics.values() if v > 0)
        
        instance_info = {
            'id': inst_id,
            'node': node,
            'metrics': metrics,
            'non_zero_count': non_zero_count
        }
        
        if non_zero_count == 0:
            without_metrics.append(instance_info)
        elif non_zero_count < 4:
            partial_metrics.append(instance_info)
        else:
            with_metrics.append(instance_info)
    
    # Report with metrics
    if with_metrics:
        print("✓ INSTANCES WITH ALL METRICS:")
        for inst in with_metrics:
            print(f"\n  {inst['id']}@{inst['node']}:")
            for k, v in inst['metrics'].items():
                print(f"    - {k}: {v:.6f}")
    
    # Report partial metrics
    if partial_metrics:
        print(f"\n⚠ INSTANCES WITH PARTIAL METRICS ({len(partial_metrics)}):")
        for inst in partial_metrics:
            print(f"\n  {inst['id']}@{inst['node']} ({inst['non_zero_count']}/4 metrics):")
            for k, v in inst['metrics'].items():
                status = "✓" if v > 0 else "✗"
                print(f"    {status} {k}: {v:.6f}")
    
    # Report without metrics
    if without_metrics:
        print(f"\n✗ INSTANCES WITHOUT METRICS ({len(without_metrics)}):")
        for inst in without_metrics:
            print(f"  - {inst['id']}@{inst['node']}")
    
    # Summary statistics
    print(f"\n=== SUMMARY ===")
    print(f"  With all metrics:     {len(with_metrics)} instances")
    print(f"  With some metrics:    {len(partial_metrics)} instances")
    print(f"  With no metrics:      {len(without_metrics)} instances")
    print(f"  Total instances:      {len(instances)}")
    
    if len(instances) > 0:
        pct_with_metrics = (len(with_metrics) / len(instances)) * 100
        print(f"  Coverage:             {pct_with_metrics:.1f}%")
    
    # Recommendations
    print(f"\n=== RECOMMENDATIONS ===\n")
    
    if without_metrics:
        print("1. CHECK: Rancher metrics are missing")
        print(f"   Run: python debug_rancher_metrics.py")
        print(f"   to see available metrics in Prometheus\n")
    
    if partial_metrics:
        print("2. CHECK: Some instances have incomplete metrics")
        print(f"   Likely causes:")
        print(f"   - CPU/RAM queries not matching your Rancher labels")
        print(f"   - exec_time not available for some workloads\n")
    
    if len(with_metrics) > 0:
        print("3. SUCCESS: Metrics are being collected")
        print(f"   {len(with_metrics)} instances have full metrics")
        print(f"   Continue monitoring by re-running this script\n")
    else:
        print("3. ACTION: No instances have complete metrics yet")
        print(f"   Run training for a few iterations:")
        print(f"   python src/train.py --episodes-per-scenario 5\n")

def main():
    if len(sys.argv) < 2:
        # Try common paths
        default_paths = [
            'src/config/services_update.xml',
            'config/services_update.xml',
            'services_update.xml',
        ]
        
        xml_path = None
        for path in default_paths:
            if Path(path).exists():
                xml_path = path
                break
        
        if not xml_path:
            print("Usage: python inspect_xml_metrics.py <path_to_services_update.xml>")
            print("\nTrying to find services_update.xml in common locations...")
            print(f"  Checked: {', '.join(default_paths)}")
            print("\n❌ File not found in any default location")
            sys.exit(1)
    else:
        xml_path = sys.argv[1]
    
    inspect_metrics_xml(xml_path)

if __name__ == '__main__':
    main()
