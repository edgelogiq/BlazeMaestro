#!/usr/bin/env python3
"""
Emulate WiFi degradation due to distance by gradually increasing latency and packet loss.

This script uses Linux traffic control (tc) to simulate WiFi signal degradation:
- Starts with baseline network conditions
- Gradually increases latency and packet loss over the execution period
- Cleans up all modifications when interrupted (Ctrl+C) or when script finishes

Requirements:
- Linux OS with tc (traffic control) available
- sudo/root access required for network interface modifications
- Network interface name (e.g., 'eth0', 'wlan0', 'docker0')

Example Usage:
    python emulate_wifi_degradation.py --interface eth0 --duration 60 \
        --start-latency 10 --end-latency 100 \
        --start-loss 0 --end-loss 5

This would:
- Gradually increase latency from 10ms to 100ms over 60 seconds
- Gradually increase packet loss from 0% to 5% over 60 seconds
- Clean up all changes when finished or interrupted
"""

import argparse
import subprocess
import time
import signal
import sys
import logging
from typing import Optional, Tuple


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class WiFiDegradationEmulator:
    """Emulates WiFi degradation by gradually increasing latency and packet loss."""

    def __init__(
        self,
        interface: str,
        duration: float,
        start_latency: float = 10.0,
        end_latency: float = 100.0,
        start_loss: float = 0.0,
        end_loss: float = 5.0,
        variation: float = 0.1,
        peak_latency: Optional[float] = None,
        peak_time: Optional[float] = None,
        drop_value: Optional[float] = None,
        drop_time: Optional[float] = None,
    ):
        """
        Initialize WiFi degradation emulator.

        Args:
            interface: Network interface name (e.g., 'eth0', 'wlan0')
            duration: Duration in seconds to run the degradation
            start_latency: Starting latency in ms
            end_latency: Ending latency in ms
            start_loss: Starting packet loss in %
            end_loss: Ending packet loss in %
            variation: Random jitter as fraction of current value (0-1)
            peak_latency: Peak latency to reach (enables three-phase mode)
            peak_time: Time in seconds to reach peak latency
            drop_value: Latency value to drop to after peak
            drop_time: Time in seconds when drop occurs
        """
        self.interface = interface
        self.duration = duration
        self.start_latency = start_latency
        self.end_latency = end_latency
        self.start_loss = start_loss
        self.end_loss = end_loss
        self.variation = variation
        self.peak_latency = peak_latency
        self.peak_time = peak_time
        self.drop_value = drop_value
        self.drop_time = drop_time
        self.start_time = None
        self.is_cleaned_up = False
        self.use_three_phase = all(
            v is not None for v in [peak_latency, peak_time, drop_value, drop_time]
        )

        # Register signal handlers for graceful cleanup
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle interrupt signals (Ctrl+C, SIGTERM)."""
        logger.info(f"\nReceived signal {signum}. Cleaning up...")
        self.cleanup()
        sys.exit(0)

    def _run_command(self, cmd: str, check: bool = True) -> Tuple[int, str]:
        """
        Run a shell command with sudo.

        Args:
            cmd: Command to run
            check: Whether to raise exception on non-zero exit

        Returns:
            Tuple of (return_code, output)
        """
        try:
            result = subprocess.run(
                f"sudo {cmd}",
                shell=True,
                capture_output=True,
                text=True,
                check=False,
            )
            return result.returncode, result.stdout + result.stderr
        except Exception as e:
            logger.error(f"Failed to run command '{cmd}': {e}")
            if check:
                raise
            return 1, str(e)

    def _verify_interface_exists(self) -> bool:
        """Verify the network interface exists."""
        returncode, output = self._run_command(
            f"ip link show {self.interface}", check=False
        )
        if returncode != 0:
            logger.error(f"Network interface '{self.interface}' not found")
            logger.info("Available interfaces:")
            _, interfaces = self._run_command("ip link show", check=False)
            logger.info(interfaces)
            return False
        return True

    def setup(self) -> bool:
        """Set up traffic control on the interface."""
        logger.info(f"Setting up traffic control on interface '{self.interface}'")

        # Check if interface exists
        if not self._verify_interface_exists():
            return False

        # Remove any existing qdisc (in case previous run was interrupted)
        logger.debug(f"Removing existing qdisc on {self.interface}...")
        self._run_command(f"tc qdisc del dev {self.interface} root", check=False)
        time.sleep(0.1)

        # Add root qdisc (HTB - Hierarchical Token Bucket)
        logger.debug(f"Adding root qdisc to {self.interface}...")
        returncode, output = self._run_command(
            f"tc qdisc add dev {self.interface} root handle 1: htb default 1"
        )
        if returncode != 0:
            logger.error(f"Failed to add root qdisc: {output}")
            return False

        # Add class for traffic shaping
        logger.debug(f"Adding HTB class...")
        returncode, output = self._run_command(
            f"tc class add dev {self.interface} parent 1: classid 1:1 htb rate 1000mbit"
        )
        if returncode != 0:
            logger.error(f"Failed to add HTB class: {output}")
            self.cleanup()
            return False

        # Add netem qdisc for latency and loss simulation
        logger.debug(f"Adding netem qdisc for network emulation...")
        returncode, output = self._run_command(
            f"tc qdisc add dev {self.interface} parent 1:1 handle 2: netem "
            f"delay {self.start_latency}ms loss {self.start_loss}%"
        )
        if returncode != 0:
            logger.error(f"Failed to add netem qdisc: {output}")
            self.cleanup()
            return False

        if self.use_three_phase:
            logger.info(
                f"✓ Traffic control setup complete on {self.interface}\n"
                f"  Mode: Three-phase (increase → drop → decrease)\n"
                f"  Initial latency: {self.start_latency}ms\n"
                f"  Peak latency: {self.peak_latency}ms (at {self.peak_time}s)\n"
                f"  Drop value: {self.drop_value}ms (at {self.drop_time}s)\n"
                f"  End latency: {self.end_latency}ms\n"
                f"  Initial packet loss: {self.start_loss}%\n"
                f"  Duration: {self.duration}s"
            )
        else:
            logger.info(
                f"✓ Traffic control setup complete on {self.interface}\n"
                f"  Mode: Linear (start → end)\n"
                f"  Initial latency: {self.start_latency}ms\n"
                f"  End latency: {self.end_latency}ms\n"
                f"  Initial packet loss: {self.start_loss}%\n"
                f"  Duration: {self.duration}s"
            )
        self.start_time = time.time()
        return True

    def _calculate_current_values(self) -> Tuple[float, float]:
        """Calculate current latency and loss based on elapsed time.
        
        Supports two modes:
        1. Simple linear (default): interpolate from start to end
        2. Three-phase: increase to peak, drop to value, gradually decrease
        """
        if self.start_time is None:
            return self.start_latency, self.start_loss

        elapsed = time.time() - self.start_time
        progress = min(elapsed / self.duration, 1.0)

        # Select latency calculation mode
        if self.use_three_phase:
            current_latency = self._calculate_three_phase_latency(elapsed)
        else:
            # Simple linear interpolation
            current_latency = self.start_latency + (
                self.end_latency - self.start_latency
            ) * progress

        # Loss always uses linear interpolation
        current_loss = self.start_loss + (self.end_loss - self.start_loss) * progress

        # Add random jitter
        if self.variation > 0:
            import random

            latency_jitter = current_latency * self.variation * (
                random.random() - 0.5
            ) * 2
            loss_jitter = current_loss * self.variation * (random.random() - 0.5) * 2

            current_latency = max(0, current_latency + latency_jitter)
            current_loss = max(0, min(100, current_loss + loss_jitter))

        return current_latency, current_loss

    def _calculate_three_phase_latency(self, elapsed: float) -> float:
        """Calculate latency for three-phase mode: increase -> drop -> decrease.
        
        Args:
            elapsed: Elapsed time in seconds
            
        Returns:
            Current latency in ms
        """
        # Phase 1: Increase from start_latency to peak_latency (0 to peak_time)
        if elapsed <= self.peak_time:
            progress = elapsed / self.peak_time
            return self.start_latency + (
                self.peak_latency - self.start_latency
            ) * progress

        # Phase 2: Drop from peak_latency to drop_value (at drop_time)
        elif elapsed <= self.drop_time:
            # Instantaneous drop (or very fast)
            return self.drop_value

        # Phase 3: Gradually decrease from drop_value to end_latency
        else:
            time_in_phase3 = elapsed - self.drop_time
            remaining_time = self.duration - self.drop_time
            
            if remaining_time <= 0:
                return self.end_latency
            
            progress = min(time_in_phase3 / remaining_time, 1.0)
            return self.drop_value + (self.end_latency - self.drop_value) * progress

    def update(self) -> bool:
        """Update network conditions based on elapsed time."""
        current_latency, current_loss = self._calculate_current_values()

        # Round to reasonable precision
        current_latency = round(current_latency, 2)
        current_loss = round(current_loss, 2)

        # Update netem qdisc
        returncode, output = self._run_command(
            f"tc qdisc change dev {self.interface} parent 1:1 handle 2: netem "
            f"delay {current_latency}ms loss {current_loss}%",
            check=False,
        )

        if returncode != 0:
            logger.error(f"Failed to update network conditions: {output}")
            return False

        return True

    def get_progress(self) -> Tuple[float, float, float]:
        """Get current progress and values."""
        if self.start_time is None:
            return 0.0, self.start_latency, self.start_loss

        elapsed = time.time() - self.start_time
        current_latency, current_loss = self._calculate_current_values()
        progress = min(elapsed / self.duration, 1.0)

        return progress, current_latency, current_loss

    def show_progress(self):
        """Display current progress and network conditions."""
        progress, latency, loss = self.get_progress()
        progress_percent = progress * 100
        bar_length = 40
        filled = int(bar_length * progress)
        bar = "█" * filled + "░" * (bar_length - filled)

        print(
            f"\r[{bar}] {progress_percent:5.1f}% | "
            f"Latency: {latency:6.2f}ms | "
            f"Loss: {loss:5.2f}% ",
            end="",
            flush=True,
        )

    def run(self):
        """Run the WiFi degradation simulation."""
        if not self.setup():
            return False

        logger.info("\nStarting WiFi degradation simulation...")
        logger.info("Press Ctrl+C to stop and clean up\n")

        update_interval = 0.5  # Update every 500ms
        last_update = time.time()

        try:
            while True:
                elapsed = time.time() - self.start_time

                # Update network conditions periodically
                if time.time() - last_update >= update_interval:
                    if not self.update():
                        logger.error("Failed to update network conditions")
                        return False
                    last_update = time.time()

                # Show progress
                self.show_progress()

                # Check if simulation is complete
                if elapsed >= self.duration:
                    logger.info("\n\n✓ WiFi degradation simulation completed")
                    break

                time.sleep(0.1)

        except KeyboardInterrupt:
            logger.info("\n\nInterrupted by user")
        finally:
            self.cleanup()

        return True

    def cleanup(self) -> bool:
        """Remove all traffic control modifications."""
        if self.is_cleaned_up:
            return True

        logger.info(f"Cleaning up traffic control on {self.interface}...")

        returncode, output = self._run_command(
            f"tc qdisc del dev {self.interface} root", check=False
        )

        if returncode != 0:
            logger.warning(
                f"Failed to completely remove traffic control: {output}\n"
                f"You may need to run: sudo tc qdisc del dev {self.interface} root"
            )
        else:
            logger.info(f"✓ Traffic control removed from {self.interface}")

        self.is_cleaned_up = True
        return returncode == 0

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Emulate WiFi degradation by gradually increasing latency and packet loss",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Simulate WiFi distance degradation from 10ms to 100ms over 60 seconds
  python emulate_wifi_degradation.py --interface eth0 --duration 60 \\
      --start-latency 10 --end-latency 100

  # Include packet loss degradation
  python emulate_wifi_degradation.py --interface wlan0 --duration 120 \\
      --start-latency 5 --end-latency 80 \\
      --start-loss 0 --end-loss 10

  # With random jitter (more realistic WiFi)
  python emulate_wifi_degradation.py --interface docker0 --duration 60 \\
      --start-latency 10 --end-latency 100 --variation 0.2
  # Three-phase mode: increase to peak, drop, then gradually decrease
  python emulate_wifi_degradation.py --interface eth0 --duration 120 \
      --start-latency 10 --peak-latency 200 --peak-time 40 \
      --drop-value 50 --drop-time 60 --end-latency 20
Requirements:
  - Linux OS with 'tc' (traffic control) command available
  - sudo/root access required
  - Network interface must exist
        """,
    )

    parser.add_argument(
        "--interface",
        required=True,
        help="Network interface name (e.g., eth0, wlan0, docker0)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=60,
        help="Duration in seconds (default: 60)",
    )
    parser.add_argument(
        "--start-latency",
        type=float,
        default=10.0,
        help="Starting latency in ms (default: 10)",
    )
    parser.add_argument(
        "--end-latency",
        type=float,
        default=100.0,
        help="Ending latency in ms (default: 100)",
    )
    parser.add_argument(
        "--start-loss",
        type=float,
        default=0.0,
        help="Starting packet loss in %% (default: 0)",
    )
    parser.add_argument(
        "--end-loss",
        type=float,
        default=5.0,
        help="Ending packet loss in %% (default: 5)",
    )
    parser.add_argument(
        "--variation",
        type=float,
        default=0.1,
        help="Random jitter as fraction of current value 0-1 (default: 0.1 = 10%% jitter)",
    )
    parser.add_argument(
        "--peak-latency",
        type=float,
        default=None,
        help="Peak latency in ms (enables three-phase mode: increase → drop → decrease)",
    )
    parser.add_argument(
        "--peak-time",
        type=float,
        default=None,
        help="Time in seconds to reach peak latency",
    )
    parser.add_argument(
        "--drop-value",
        type=float,
        default=None,
        help="Latency value to drop to after reaching peak",
    )
    parser.add_argument(
        "--drop-time",
        type=float,
        default=None,
        help="Time in seconds when the drop to drop-value occurs",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose debug logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    emulator = WiFiDegradationEmulator(
        interface=args.interface,
        duration=args.duration,
        start_latency=args.start_latency,
        end_latency=args.end_latency,
        start_loss=args.start_loss,
        end_loss=args.end_loss,
        variation=args.variation,
        peak_latency=args.peak_latency,
        peak_time=args.peak_time,
        drop_value=args.drop_value,
        drop_time=args.drop_time,
    )

    success = emulator.run()
    sys.exit(0 if success else 1)

# python emulate_wifi_degradation.py --interface eth0 --duration 120 \
#     --start-latency 10 --peak-latency 200 --peak-time 40 \
#     --drop-value 50 --drop-time 60 --end-latency 10
if __name__ == "__main__":
    main()
