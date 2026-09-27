#!/bin/bash
# Quick demonstration of robot movement simulation script

echo "=================================="
echo "Robot Movement Simulation Demo"
echo "=================================="
echo ""
echo "This script demonstrates the robot movement simulation"
echo "with different scenarios and parameters."
echo ""

# Quick 2-minute test
echo "1. Running quick 2-minute test..."
python scripts/simulate_robot_movement.py \
  --duration 120 \
  --speed 1.0 \
  --report-interval 20 \
  2>&1 | grep -E "Time\(s\)|SIMULATION|SNR CHANGE|Events tracked"

echo ""
echo "✓ Quick test complete!"
echo ""

# Default scenario (10 minutes)
echo "2. Running full default scenario (600 seconds)..."
python scripts/simulate_robot_movement.py \
  --duration 600 \
  --speed 0.33 \
  --report-interval 60 \
  2>&1 | grep -E "Time\(s\)|SIMULATION|SNR CHANGE|Events tracked"

echo ""
echo "✓ Full scenario complete!"
echo ""

# Fast robot scenario
echo "3. Running fast robot scenario (300 seconds)..."
python scripts/simulate_robot_movement.py \
  --duration 300 \
  --speed 0.5 \
  --report-interval 30 \
  2>&1 | grep -E "Time\(s\)|SIMULATION|SNR CHANGE|Events tracked"

echo ""
echo "✓ Fast robot scenario complete!"
echo ""

echo "=================================="
echo "All simulations complete!"
echo "=================================="
echo ""
echo "Check the output in: out/simulation/"
echo "  - robot_movement_events.json"
echo "  - robot_movement_report.txt"
echo ""
