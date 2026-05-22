#!/usr/bin/env python3
import time
import json
import random
import math
import argparse
from typing import Dict, List, Any

def run_benchmark_trials(
    trials: int = 200,
    rtt_ms: float = 20.0,
    packet_loss: float = 0.0,
    bandwidth_mbps: float = 100.0,
    perception_mode: str = "A",
    compute_mode: int = 3
) -> Dict[str, Any]:
    """
    Runs a series of simulated trials representing VICKY system cycles.
    Injects network degradation parameters to measure latency, Safety Reaction Time (SRT),
    depth resolution error, and failure/hallucination metrics.
    """
    print("=" * 65)
    print(f"      STARTING VICKY BENCHMARK HARNESS ({trials} TRIALS)")
    print("=" * 65)
    print(f"  Configuration:")
    print(f"    Compute Node Mode:   Approach {compute_mode}")
    print(f"    Perception Paradigm: Approach {perception_mode}")
    print(f"    Network Bandwidth:   {bandwidth_mbps} Mbps")
    print(f"    Network Base RTT:    {rtt_ms} ms")
    print(f"    Packet Loss Rate:    {packet_loss * 100.0:.1f}%")
    print("-" * 65)

    srt_list = []
    yolo_latencies = []
    reasoning_latencies = []
    network_delays = []
    depth_errors = []
    
    timeouts = 0
    missing_targets = 0
    hallucinations = 0
    successful_runs = 0
    
    # Ground truth: An obstacle is placed at exactly 1.85 meters depth directly center
    ground_truth_depth_meters = 1.85
    
    # Bandwidth conversion to bytes per millisecond
    bandwidth_bytes_ms = (bandwidth_mbps * 1e6 / 8) / 1000.0
    # Average compressed JPEG frame size (672x376 @ 75% quality is ~45KB)
    frame_size_bytes = 45000.0
    
    for i in range(1, trials + 1):
        # 1. Frame Capture (ZED local camera timing)
        t_capture = random.uniform(2.0, 5.0) # 2-5ms frame capture overhead
        
        # 2. Network Transmission Latency Simulation
        # Delay = base RTT / 2 (one-way) + serialization time based on bandwidth
        trans_delay_ms = (rtt_ms / 2.0) + (frame_size_bytes / bandwidth_bytes_ms)
        
        # Handle packet drop simulation
        is_dropped = random.random() < packet_loss
        retransmission_overhead_ms = 0.0
        if is_dropped:
            # Retransmission triggers after TCP timeout (typically RTT + 200ms)
            retransmission_overhead_ms = rtt_ms + 200.0
            
        total_network_delay_ms = trans_delay_ms + retransmission_overhead_ms
        network_delays.append(total_network_delay_ms)
        
        # 3. Model Inference Latency Simulation
        t_detect = 0.0
        t_process = 0.0
        
        if perception_mode == "A":
            # YOLOv8 Detection Latency
            t_detect = random.uniform(12.0, 18.0) if compute_mode == 1 else random.uniform(8.0, 12.0)
            # Ollama Phi-3 System reasoning
            t_process = random.uniform(40.0, 75.0)
        else:
            # End-to-end Visual VLM (Qwen2-VL / SmolVLM)
            t_detect = 0.0
            t_process = random.uniform(250.0, 480.0)
            
        yolo_latencies.append(t_detect)
        reasoning_latencies.append(t_process)
        
        # 4. Local Command Generation & TTS Playback Overhead
        t_command = random.uniform(5.0, 12.0)
        t_actuate = random.uniform(50.0, 100.0) # User audio reaction response loop delay
        
        # 5. Calculate Total Safety Reaction Time (SRT)
        # For Local Mode (Approach 1): No network delay
        # For Cloud Mode (Approach 2): Full network delay for frames & response
        # For Hybrid Mode (Approach 3): Local safety loop bypasses network; verbal queries use network
        if compute_mode == 1:
            srt = t_capture + t_detect + t_process + t_command + t_actuate
        elif compute_mode == 2:
            srt = t_capture + (total_network_delay_ms * 2.0) + t_detect + t_process + t_command + t_actuate
        else: # Hybrid
            # Emergency Stop SRT is local, but verbal scene description SRT includes cloud network
            srt = t_capture + t_detect + t_command + t_actuate # Local immediate safety reaction
            
        srt_list.append(srt)
        
        # 6. Depth Resolution Error Simulation
        # Simulate slight Gaussian noise in ZED depth metrics estimation
        computed_depth = ground_truth_depth_meters + random.normalvariate(0.0, 0.03) # 3cm std dev
        depth_err = abs(computed_depth - ground_truth_depth_meters)
        depth_errors.append(depth_err)
        
        # 7. Failures and Hallucinations mapping
        # Timeout occurs if network delays exceed a 1.0-second safety threshold
        if compute_mode in [2, 3] and total_network_delay_ms > 1000.0:
            timeouts += 1
            
        # Target missing due to bounding box detector misclassifications (false negatives)
        if perception_mode == "A" and random.random() < 0.02: # 2% miss rate
            missing_targets += 1
            
        # Hallucination occurs in generative language reasoning (VLM has higher rates)
        hallucination_rate = 0.08 if perception_mode == "B" else 0.015
        if random.random() < hallucination_rate:
            hallucinations += 1
        else:
            successful_runs += 1
            
    # Calculate Statistical Aggregates
    mean_srt = sum(srt_list) / trials
    std_srt = math.sqrt(sum((x - mean_srt) ** 2 for x in srt_list) / trials)
    
    mean_yolo = sum(yolo_latencies) / trials
    mean_reason = sum(reasoning_latencies) / trials
    mean_net = sum(network_delays) / trials
    mae_depth = sum(depth_errors) / trials
    
    success_rate = (successful_runs / trials) * 100.0
    
    print("-" * 65)
    print("      BENCHMARK ANALYSIS COMPLETE - TRIAL RESULTS")
    print("-" * 65)
    print(f"  Safety Reaction Time (SRT):")
    print(f"    Mean SRT:             {mean_srt:.2f} ms")
    print(f"    Std Dev SRT:          {std_srt:.2f} ms")
    print(f"    Min SRT:              {min(srt_list):.2f} ms")
    print(f"    Max SRT:              {max(srt_list):.2f} ms")
    print(f"  Latency breakdown:")
    print(f"    Avg YOLO Latency:     {mean_yolo:.2f} ms")
    print(f"    Avg LLM/VLM Latency:  {mean_reason:.2f} ms")
    print(f"    Avg Network Delay:    {mean_net:.2f} ms")
    print(f"  Spatial Accuracy:")
    print(f"    Depth Resolution MAE: {mae_depth * 100.0:.2f} cm (True target: {ground_truth_depth_meters}m)")
    print(f"  System Failures:")
    print(f"    Timeouts (>1000ms):   {timeouts}")
    print(f"    Missing Targets:      {missing_targets}")
    print(f"    Hallucinations:       {hallucinations}")
    print(f"    Overall Success Rate: {success_rate:.1f}%")
    print("=" * 65)

    return {
        "mean_srt_ms": mean_srt,
        "std_srt_ms": std_srt,
        "mean_yolo_ms": mean_yolo,
        "mean_reason_ms": mean_reason,
        "mean_net_ms": mean_net,
        "depth_mae_cm": mae_depth * 100.0,
        "timeouts": timeouts,
        "missing_targets": missing_targets,
        "hallucinations": hallucinations,
        "success_rate": success_rate
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VICKY Project Benchmark Harness")
    parser.add_argument("--trials", type=int, default=200, help="Number of benchmark trials")
    parser.add_argument("--rtt", type=float, default=30.0, help="Simulated network RTT overhead in ms")
    parser.add_argument("--loss", type=float, default=0.02, help="Simulated packet drop rate (0.0 to 1.0)")
    parser.add_argument("--bandwidth", type=float, default=25.0, help="Simulated network bandwidth in Mbps")
    parser.add_argument("--perception", type=str, choices=["A", "B"], default="A", help="Perception Paradigm (A or B)")
    parser.add_argument("--mode", type=int, choices=[1, 2, 3], default=3, help="Compute Mode (1, 2, or 3)")
    
    args = parser.parse_args()
    
    run_benchmark_trials(
        trials=args.trials,
        rtt_ms=args.rtt,
        packet_loss=args.loss,
        bandwidth_mbps=args.bandwidth,
        perception_mode=args.perception,
        compute_mode=args.mode
    )
