#!/usr/bin/env python3
"""
Fast Robot Movement Simulation - Guaranteed to show SNR changes

This version uses aggressive parameters to ensure visible WiFi quality changes
and reallocation events in a short time.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import logging
from EdgeLogiQ.BlazeAIoT.BlazeMaestro.Utils.wifi_mobility_controller import WiFiMobilityController

# Setup logging
logging.basicConfig(
    level=logging.WARNING,  # Less verbose
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("FastRobotSim")
logger.setLevel(logging.INFO)

def main():
    print("\n" + "="*80)
    print("FAST ROBOT MOVEMENT SIMULATION")
    print("Demonstrating WiFi quality changes and reallocation triggers")
    print("="*80 + "\n")
    
    # Initialize WiFi controller
    wifi = WiFiMobilityController()
    
    # Setup infrastructure
    logger.info("Setting up WiFi infrastructure...")
    wifi.register_access_point("edge-gpu1", x=0.0, y=0.0, coverage_radius=50.0)
    wifi.register_access_point("edge-gpu2", x=100.0, y=0.0, coverage_radius=50.0)
    wifi.register_robot("frank-1", x=5.0, y=5.0, vx=0.0, vy=0.0)
    wifi.register_wifi_link("frank-1-wifi6", "frank-1",
                           bandwidths=["600Mbps", "300Mbps", "100Mbps"],
                           latencies=[0.005, 0.010, 0.020])
    
    # Start robot moving FAST
    logger.info("Starting robot movement (2.0 m/s - very fast!)")
    wifi.set_robot_velocity("frank-1", vx=2.0, vy=0.0)
    
    print("\n{:>8s} | {:>8s} | {:>10s} | {:>10s} | {:>12s} | {:>15s}".format(
        "Time(s)", "Distance", "SNR", "Bandwidth", "Latency", "Connected AP"
    ))
    print("-" * 90)
    
    current_time = 0.0
    last_snr = None
    snr_changes = 0
    
    # Run for 60 seconds with fast robot
    while current_time <= 60.0:
        # Update WiFi controller
        updates = wifi.update(current_time)
        
        # Get robot status
        status = wifi.get_robot_status("frank-1")
        link = wifi.get_link_characteristics("frank-1-wifi6")
        
        if not status or not link:
            current_time += 1.0
            continue
        
        # Calculate distance to nearest AP
        pos = status['position']
        dist_ap1 = ((pos['x'] - 0.0)**2 + (pos['y'] - 0.0)**2)**0.5
        dist_ap2 = ((pos['x'] - 100.0)**2 + (pos['y'] - 0.0)**2)**0.5
        min_dist = min(dist_ap1, dist_ap2)
        
        # Track SNR changes
        current_snr = link.get('snr_level', 0)
        bandwidth = link.get('bandwidth', 'N/A')
        latency = link.get('latency', 0.0)
        connected_ap = status.get('connected_ap', 'none')
        if last_snr is not None and current_snr != last_snr:
            snr_changes += 1
            print(f"\n⚠️  SNR CHANGE at t={current_time:.1f}s: Level {last_snr} → {current_snr}")
            print(f"    This triggers a potential SERVICE REALLOCATION!")
            print()
        
        # Print status every second
        if int(current_time) % 5 == 0 or updates:
            snr_names = {0: "GOOD", 1: "MEDIUM", 2: "POOR"}
            bw_str = str(bandwidth) if bandwidth else "N/A"
            lat_ms = (latency * 1000) if latency else 0.0
            ap_str = str(connected_ap) if connected_ap else "none"
            
            print("{:8.1f} | {:8.1f}m | {:>10s} | {:>10s} | {:9.1f}ms | {:>15s}".format(
                current_time,
                min_dist,
                snr_names.get(current_snr, "UNK"),
                bw_str,
                lat_ms,
                ap_str
            ))
        
        last_snr = current_snr
        current_time += 1.0
    
    print("\n" + "="*80)
    print("SIMULATION SUMMARY")
    print("="*80)
    print(f"Total SNR changes (reallocation triggers): {snr_changes}")
    print(f"Final robot position: ({status['position']['x']:.1f}, {status['position']['y']:.1f})")
    print(f"Distance traveled: {status['position']['x'] - 5.0:.1f} meters")
    print("="*80 + "\n")
    
    if snr_changes == 0:
        print("⚠️  No SNR changes detected!")
        print("Try increasing simulation duration or robot speed.")
    else:
        print(f"✅ Success! Detected {snr_changes} reallocation opportunities!")
        print("\nEach SNR change represents a point where the orchestrator would:")
        print("  • Evaluate current service placement")
        print("  • Consider network latency impact")
        print("  • Potentially migrate services between robot and edge")
        print("  • Balance battery usage vs. service quality")

if __name__ == "__main__":
    main()
