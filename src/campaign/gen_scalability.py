"""Synthetic platform/service XML generator for the scalability study (R2-11).

Produces a platform XML (SimGrid format, same props as src/config/platform_sim.xml) and a
services XML (same schema as src/config/services_sim.xml) controlled by:

- n_nodes      schedulable compute nodes (robots excluded)
- n_services   total services, including the per-robot core graph
- n_robots     battery-powered robots, each behind its own WiFi link
- n_clusters   network sites (routers); layers cycle edge, fog, edge, cloud, ... so that
               every platform has at least one site of each layer when n_clusters >= 3
- depth        maximum depth of the extra service trees hung under each robot's graph

Per robot, the core graph mirrors the testbed application: a node-locked driver
(robot-k), slam-k <- robot-k, localization-k <- robot-k, estop-k (GPU, soft dependency).
The remaining services form random trees of blocking dependencies rooted at slam-k or
localization-k, up to ``depth`` levels. Execution times scale with node speed; GPU
services only have measurements on GPU nodes (elsewhere they fall back to a 10x slower
container estimate, so a CPU placement violates their period, as for estop on the testbed).

Usage:
    python src/campaign/gen_scalability.py --nodes 25 --services 50 --robots 2 \
        --clusters 3 --depth 4 --seed 1 --out out/campaign/scal/inst/n25_s50
"""

import argparse
import os
import random
from xml.sax.saxutils import quoteattr as q

LAYER_CYCLE = ["edge", "fog", "edge", "cloud"]
CORE_LAT_MS = {"edge": 5.0, "fog": 2.0, "cloud": 45.0}  # router <-> core backbone
INTERNAL_LAT_MS = {"edge": 3.0, "fog": 3.0, "cloud": 2.0}
REF_SPEED_GF = 40.0


def make_platform(n_nodes, n_robots, n_clusters, rng):
    layers = [LAYER_CYCLE[i % len(LAYER_CYCLE)] for i in range(max(1, n_clusters))]
    if n_clusters >= 3 and "cloud" not in layers:
        layers[-1] = "cloud"
    sites = [f"site{i}-{layer}" for i, layer in enumerate(layers)]
    hosts = []  # (id, layer, site, cores, ram, gpu_gb, speed)
    for i in range(n_nodes):
        s = i % len(sites)
        layer = layers[s]
        gpu = rng.random() < 0.3 or i == 0  # at least one GPU node
        cores = rng.choice([4, 8, 8, 16] if layer != "cloud" else [8, 16, 16, 32])
        ram = rng.choice([8.0, 15.5, 15.5, 31.0])
        speed = rng.uniform(20, 40) if layer == "edge" else rng.uniform(35, 60) if layer == "fog" else rng.uniform(60, 120)
        hosts.append((f"{layer}-n{i}", layer, sites[s], cores, ram, rng.choice([4.0, 8.0]) if gpu else 0.0, speed))
    edge_sites = [s for s, layer in zip(sites, layers) if layer == "edge"] or sites
    robots = [(f"robot-{k}", edge_sites[k % len(edge_sites)]) for k in range(n_robots)]

    out = ['<?xml version="1.0" encoding="UTF-8"?>',
           '<!DOCTYPE platform SYSTEM "https://simgrid.org/simgrid.dtd">',
           '<platform version="4.1">', '  <zone id="world" routing="Floyd">']
    for hid, layer, _, cores, ram, gpu, speed in hosts:
        out.append(f'    <host id={q(hid)} speed="{speed:.0f}Gf" core="{cores}">')
        out.append(f'      <prop id="ram.GB" value="{ram}" />')
        if gpu:
            out.append('      <prop id="gpu.count" value="1" />')
            out.append(f'      <prop id="gpu.memGB" value="{gpu:.0f}" />')
        out.append(f'      <prop id="layer" value="{layer}" />')
        out.append('      <prop id="schedulable" value="true" />')
        out.append('    </host>')
    for rid, _ in robots:
        out += [f'    <host id={q(rid)} speed="15Gf" core="8">',
                '      <prop id="ram.GB" value="30.0" />',
                '      <prop id="battery.powered" value="true" />',
                '      <prop id="battery.voltage" value="12.0" />',
                '      <prop id="battery.capacity.ah" value="21.0" />',
                '      <prop id="power.jetson.base" value="5.0" />',
                '      <prop id="power.jetson.active" value="40.0" />',
                '      <prop id="power.motor.max" value="24.0" />',
                '      <prop id="layer" value="edge" />',
                '      <prop id="gpu.count" value="1" />',
                '      <prop id="gpu.memGB" value="8" />',
                '      <prop id="schedulable" value="true" />',
                '    </host>']
    out.append('    <router id="core-router" />')
    for s in sites:
        out.append(f'    <router id="{s}-router" />')
    for s, layer in zip(sites, layers):
        out.append(f'    <link id="{s}-internal" bandwidth="1Gbps" latency="{INTERNAL_LAT_MS[layer]}ms" />')
        out.append(f'    <link id="{s}-uplink" bandwidth="1Gbps" latency="{CORE_LAT_MS[layer]}ms" />')
    for rid, _ in robots:
        out.append(f'    <link id="{rid}-wifi6" bandwidth="600Mbps" latency="5ms" />')
    for hid, _, site, *_ in hosts:
        out += [f'    <route src={q(hid)} dst="{site}-router" symmetrical="YES">',
                f'      <link_ctn id="{site}-internal" />', '    </route>']
    for rid, site in robots:
        out += [f'    <route src={q(rid)} dst="{site}-router" symmetrical="YES">',
                f'      <link_ctn id="{rid}-wifi6" />', '    </route>']
    for s in sites:
        out += [f'    <route src="{s}-router" dst="core-router" symmetrical="YES">',
                f'      <link_ctn id="{s}-uplink" />', '    </route>']
    out += ['  </zone>', '</platform>']
    return "\n".join(out) + "\n", hosts, robots


def make_services(n_services, robots, hosts, depth, rng):
    all_nodes = [(h[0], h[1], h[5] > 0, h[6]) for h in hosts] + [(r, "edge", True, 15.0) for r, _ in robots]
    services = []  # dicts

    def svc(sid, rate, exec_ref, cpu, ram, deps, gpu=False, lock=None):
        services.append(dict(id=sid, rate=rate, exec_ref=exec_ref, cpu=cpu, ram=ram, deps=deps, gpu=gpu, lock=lock))

    for rid, _ in robots:
        k = rid.split("-")[1]
        svc(f"robot-{k}-driver", 7.5, 0.014, 0.6, 1.2,
            [(f"estop-{k}", dict(blocking="false", feedback="true", constraint_level="soft"))], lock=rid)
        svc(f"slam-{k}", 7.5, 0.05, 0.4, 0.2, [(f"robot-{k}-driver", {})])
        svc(f"localization-{k}", 20.0, 0.008, 0.3, 0.1, [(f"robot-{k}-driver", {})])
        svc(f"estop-{k}", 15.0, 0.02, 0.9, 0.5,
            [(f"robot-{k}-driver", dict(blocking="false", size_mb="0.5", constraint_level="soft"))], gpu=True)
    # Extra services: random trees under slam/localization of a random robot.
    frontier = [(s["id"], 1) for s in services if s["id"].startswith(("slam-", "localization-"))]
    i = 0
    while len(services) < n_services:
        parent, level = rng.choice(frontier)
        sid = f"svc-{i}"
        rate = rng.choice([2.0, 5.0, 10.0])
        svc(sid, rate, rng.uniform(0.005, 0.04), rng.uniform(0.1, 0.6), rng.uniform(0.1, 0.8),
            [(parent, {})], gpu=rng.random() < 0.15)
        if level + 1 <= depth:
            frontier.append((sid, level + 1))
        i += 1

    out = ["<?xml version='1.0' encoding='utf-8'?>", "<services>"]
    for s in services[:max(n_services, 4 * len(robots))]:
        out.append(f'  <service id={q(s["id"])} metrics_service_name={q(s["id"])}>')
        if s["lock"]:
            out.append(f'    <placement type="node_lock" node={q(s["lock"])} />')
        else:
            out.append('    <placement type="flexible" />')
        if s["deps"]:
            out.append('    <dependencies>')
            for dep, attrs in s["deps"]:
                extra = "".join(f' {k}="{v}"' for k, v in attrs.items())
                out.append(f'      <dependency service={q(dep)}{extra} />')
            out.append('    </dependencies>')
        fallback = s["exec_ref"] * (10.0 if s["gpu"] else 1.5)
        out.append('    <containers>')
        out.append(f'      <container id="{s["id"]}-main" cpu="{s["cpu"]:.3f}" ram_gb="{s["ram"]:.3f}" '
                   f'exec_time="{fallback:.6f}" />')
        out.append('    </containers>')
        out.append('    <instances>')
        for nid, layer, has_gpu, speed in all_nodes:
            if s["lock"] and nid != s["lock"]:
                continue
            if s["gpu"] and not has_gpu:
                continue
            et = s["exec_ref"] * REF_SPEED_GF / speed * rng.uniform(0.9, 1.1)
            out.append(f'      <container pod_name="{s["id"]}-{nid}" node={q(nid)} layer="{layer}" '
                       f'cpu_usage_min="{0.5 * s["cpu"]:.3f}" cpu_usage_max="{s["cpu"]:.3f}" cpu_usage_last="{s["cpu"]:.3f}" '
                       f'ram_usage_min_gb="{s["ram"]:.3f}" ram_usage_max_gb="{s["ram"]:.3f}" ram_usage_last_gb="{s["ram"]:.3f}" '
                       f'exec_time="{et:.6f}" execution_rate_hz="{s["rate"]:.2f}" />')
        out.append('    </instances>')
        out.append('  </service>')
    out.append("</services>")
    return "\n".join(out) + "\n", len(services)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nodes", type=int, required=True)
    ap.add_argument("--services", type=int, required=True)
    ap.add_argument("--robots", type=int, default=1)
    ap.add_argument("--clusters", type=int, default=3)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True, help="output directory")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    os.makedirs(a.out, exist_ok=True)
    plat, hosts, robots = make_platform(a.nodes, a.robots, a.clusters, rng)
    svcs, n = make_services(a.services, robots, hosts, a.depth, rng)
    open(os.path.join(a.out, "platform.xml"), "w").write(plat)
    open(os.path.join(a.out, "services.xml"), "w").write(svcs)
    print(f"{a.out}: {a.nodes} nodes + {a.robots} robots, {a.clusters} sites, {n} services")


if __name__ == "__main__":
    main()
