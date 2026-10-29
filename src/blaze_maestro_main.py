#!/usr/bin/env python3
"""
BlazeAIoT Maestro - Main Entry Point
"""

from EdgeLogiQ.BlazeAIoT.BlazeMaestro.blaze_maestro_app import BlazeMaestroApp

import sys
import os
import argparse
import signal
import time


# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


async def main():
    parser = argparse.ArgumentParser(
        description="BlazeAIoT Maestro - Kubernetes-aware Edge Orchestrator"
    )

    parser.add_argument('--layer_config_path', type=str,
                            default='/config/layer/config.json',
                            help='Path to the configuration file')
    parser.add_argument('--node_config_path', type=str,
                        default='/config/node/config.json',
                        help='Path to the configuration file')
    parser.add_argument('--service_config_path', type=str,
                        default='/config/p3at-robot/config.json',
                        help='Path to the configuration file')
    parser.add_argument('--logfile', type=str,
                        default='/edgelogiq/logs/robot/p3at-robot/log.blz',
                        help='Path to the log file')
    
    parser.add_argument(
        '--update-configs',
        action='store_true',
        help="Update platform.xml and services.xml with Prometheus metrics before running"
    )
    parser.add_argument(
        '--update-configs-only',
        action='store_true',
        help="Update configs from Prometheus and exit (don't run orchestration)"
    )
    args = parser.parse_args()

    # Create and run Maestro
    maestro = BlazeMaestroApp(args)

    # Setup signal handlers for graceful shutdown
    def signal_handler(sig, frame):
        from BlazeAIoT.Core.Utils import BlazeLogger
        BlazeLogger.get_instance().info("Signal %s caught, exiting..." % sig)
        maestro.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Initialize and run
    try:
        await maestro.start()

        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        from BlazeAIoT.Core.Utils import BlazeLogger
        BlazeLogger.get_instance().info("KeyboardInterrupt caught, exiting...")
        sys.exit(0)
    except Exception as e:
        print(f"\n Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
