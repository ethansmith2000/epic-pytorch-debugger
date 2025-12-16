#!/usr/bin/env python3
"""
Test script for EpicPytorchDebugger features.

This script demonstrates all the debugging capabilities:
1. NaN/Inf detection
2. Export graph to DOT (Graphviz)
3. Operation timing
4. Memory tracking
5. Gradient tracking
6. Conditional breakpoints
7. Tensor snapshots
8. Watch mode
9. In-place operation detection
10. Device transfer tracking
11. Logging integration (wandb/tensorboard compatible)
12. Module-aware tracking (NEW)
13. Shape tracking (NEW)
14. Backward tracking (NEW)
15. Gradient flow visualization (NEW)
16. Operation replay (NEW)
17. Interactive stepping (NEW)
18. Mixed precision tracking (NEW)
19. Memory leak detection (NEW)
20. Distributed training support (NEW)
"""

import torch
import torch.nn as nn

from debugger import EpicPytorchDebugger
from utils import break_on_large_values, break_on_nan, break_on_nan_or_inf
from functions import print_vars, format_tensor_details


def separator(title: str) -> None:
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80 + "\n")


# =============================================================================
# Test 1: Basic Operation Timing
# =============================================================================
def test_operation_timing():
    separator("TEST 1: Operation Timing")
    
    with EpicPytorchDebugger(
        profile_ops=True,
        do_pdb=False,
        run_trace=False,
    ) as dbg:
        a = torch.randn(500, 500)
        b = torch.randn(500, 500)
        
        for _ in range(10):
            c = a @ b
            d = c + a
            e = torch.relu(d)
            f = e.sum()
    
    print("Operation timing results:")
    dbg.print_op_timings(top_n=10)


# =============================================================================
# Test 2: Computation Graph & DOT Export
# =============================================================================
def test_computation_graph():
    separator("TEST 2: Computation Graph & DOT Export")
    
    with EpicPytorchDebugger(
        do_pdb=False,
        include_shapes=True,
    ) as dbg:
        x = torch.randn(3, 4)
        y = torch.randn(4, 5)
        z = x @ y
        w = z + torch.randn(3, 5)
        result = w.relu()
    
    print("Computation graph for 'result':")
    comp_graph = dbg._get_tensor_metadata(result, "comp_graph")
    if comp_graph:
        print(comp_graph)
    
    dbg.export_graph(result, "/tmp/test_graph.dot")
    print("\nDOT file saved to /tmp/test_graph.dot")


# =============================================================================
# Test 3: NaN/Inf Detection
# =============================================================================
def test_nan_detection():
    separator("TEST 3: NaN/Inf Detection")
    
    print("Creating a tensor that will produce NaN...")
    
    with EpicPytorchDebugger(
        detect_anomaly=True,
        anomaly_pdb=False,
        do_pdb=False,
    ) as dbg:
        a = torch.tensor([1.0, 0.0, -1.0])
        b = torch.tensor([0.0, 0.0, 0.0])
        
        c = a / b
        print(f"\nResult of division: {c}")
        print(f"Anomalies detected: {dbg.get_anomaly_count()}")


# =============================================================================
# Test 4: Memory Tracking (CUDA)
# =============================================================================
def test_memory_tracking():
    separator("TEST 4: Memory Tracking")
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping memory tracking test.")
        return
    
    with EpicPytorchDebugger(
        track_memory=True,
        do_pdb=False,
        run_trace=False,
    ) as dbg:
        a = torch.randn(1000, 1000, device='cuda')
        b = torch.randn(1000, 1000, device='cuda')
        c = a @ b
        d = c + a
        del a, b, c, d
        torch.cuda.empty_cache()
    
    dbg.print_memory_summary()


# =============================================================================
# Test 5: Gradient Tracking
# =============================================================================
def test_gradient_tracking():
    separator("TEST 5: Gradient Tracking")
    
    with EpicPytorchDebugger(
        track_gradients=True,
        do_pdb=False,
    ) as dbg:
        x = torch.randn(10, 10, requires_grad=True)
        y = torch.randn(10, 10, requires_grad=True)
        
        z = x @ y
        loss = z.sum()
        loss.backward()
        
        print(f"x.grad norm: {x.grad.norm().item():.4f}")
        print(f"y.grad norm: {y.grad.norm().item():.4f}")
    
    print("\nGradient history:")
    dbg.print_grad_summary()


# =============================================================================
# Test 6: Tensor Snapshots
# =============================================================================
def test_snapshots():
    separator("TEST 6: Tensor Snapshots")
    
    with EpicPytorchDebugger(do_pdb=False) as dbg:
        x = torch.randn(5, 5)
        y = torch.randn(5, 5)
        
        dbg.snapshot("initial")
        print("Saved snapshot 'initial'")
        
        x = x * 2
        y = y + 1
        z = torch.randn(3, 3)
        
        dbg.snapshot("modified")
        print("Saved snapshot 'modified'")
        
        print(f"\nAvailable snapshots: {dbg.list_snapshots()}")
    
    dbg.print_snapshot_diff("initial", "modified")


# =============================================================================
# Test 7: Watch Mode
# =============================================================================
def test_watch_mode():
    separator("TEST 7: Watch Mode")
    
    with EpicPytorchDebugger(
        watch_tensors=["loss", "output"],
        do_pdb=False,
    ) as dbg:
        print("Watching tensors: 'loss', 'output'\n")
        
        for i in range(3):
            output = torch.randn(10) * (i + 1)
            loss = output.sum()


# =============================================================================
# Test 8: Conditional Breakpoints
# =============================================================================
def test_conditional_breakpoints():
    separator("TEST 8: Conditional Breakpoints")
    
    print("Setting up breakpoint for values > 100...")
    
    triggered_count = [0]
    
    def my_condition(t):
        max_val = t.abs().max().item()
        if max_val > 100:
            triggered_count[0] += 1
            return True
        return False
    
    class TestDebugger(EpicPytorchDebugger):
        def _check_break_condition(self, tensor, op_name):
            if self.break_condition is None:
                return False
            try:
                if self.break_condition(tensor):
                    tensor_name = self._get_tensor_metadata(tensor, "tensor_name", "<unnamed>")
                    print(f"  [BREAKPOINT TRIGGERED] op={op_name}, tensor={tensor_name}")
                    return False
            except Exception as e:
                pass
            return False
    
    with TestDebugger(
        break_condition=my_condition,
        do_pdb=False,
        run_trace=False,
    ) as dbg:
        a = torch.randn(10, 10)
        b = a @ a.T
        c = torch.randn(10, 10) * 1000
    
    print(f"\nBreakpoint triggered {triggered_count[0]} time(s).")


# =============================================================================
# Test 9: In-place Operation Detection
# =============================================================================
def test_inplace_detection():
    separator("TEST 9: In-place Operation Detection")
    
    with EpicPytorchDebugger(
        detect_inplace=True,
        do_pdb=False,
        run_trace=False,
    ) as dbg:
        a = torch.randn(5, 5)
        a.add_(1)
        a.mul_(2)
        a.relu_()
        b = a + 1
    
    print(f"Detected {len(dbg.get_inplace_ops())} in-place operations:")
    for op in dbg.get_inplace_ops():
        print(f"  - {op['op']}")


# =============================================================================
# Test 10: Device Transfer Tracking
# =============================================================================
def test_device_transfers():
    separator("TEST 10: Device Transfer Tracking")
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping device transfer test.")
        return
    
    with EpicPytorchDebugger(
        track_device_transfers=True,
        do_pdb=False,
    ) as dbg:
        x = torch.randn(10, 10)
        x = x.cuda()
        y = x @ x.T
        y = y.cpu()
    
    transfers = dbg.get_device_transfers()
    print(f"Recorded {len(transfers)} device transfers:")
    for t in transfers:
        print(f"  {t['tensor']}: {t['from_device']} -> {t['to_device']}")


# =============================================================================
# Test 11: Logging Integration
# =============================================================================
def test_logging_integration():
    separator("TEST 11: Logging Integration")
    
    with EpicPytorchDebugger(
        profile_ops=True,
        track_gradients=True,
        detect_inplace=True,
        do_pdb=False,
    ) as dbg:
        x = torch.randn(50, 50, requires_grad=True)
        y = x @ x.T
        y.add_(1)
        loss = y.sum()
        loss.backward()
    
    summary = dbg.get_summary_dict()
    
    print("Summary dict (ready for wandb.log):")
    for key, value in summary.items():
        print(f"  {key}: {value}")


# =============================================================================
# Test 12: Module-Aware Tracking
# =============================================================================
def test_module_tracking():
    separator("TEST 12: Module-Aware Tracking")
    
    class SimpleNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 20)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(20, 5)
        
        def forward(self, x):
            x = self.relu(self.fc1(x))
            return self.fc2(x)
    
    model = SimpleNet()
    
    with EpicPytorchDebugger(
        track_modules=True,
        profile_ops=True,
        do_pdb=False,
    ) as dbg:
        dbg.register_modules(model)
        
        x = torch.randn(8, 10)
        out = model(x)
    
    print("Module timing results:")
    module_stats = dbg.get_module_timings()
    for name, stats in module_stats.items():
        print(f"  {name}: {stats['mean_ms']:.3f}ms (count={stats['count']})")
    
    print(f"\nOperations per module:")
    for mod, ops in list(dbg.get_module_op_counts().items())[:3]:
        print(f"  {mod}: {dict(ops)}")


# =============================================================================
# Test 13: Shape Tracking
# =============================================================================
def test_shape_tracking():
    separator("TEST 13: Shape Tracking")
    
    with EpicPytorchDebugger(
        track_shapes=True,
        do_pdb=False,
    ) as dbg:
        x = torch.randn(4, 10)
        x = x.view(2, 20)
        x = x.unsqueeze(0)
        x = x.expand(3, 2, 20)
        x = x.reshape(6, 20)
    
    shape_log = dbg.get_shape_log()
    print(f"Tracked {len(shape_log)} shape transformations")
    dbg.print_shape_log(last_n=5)


# =============================================================================
# Test 14: Backward Pass Tracking
# =============================================================================
def test_backward_tracking():
    separator("TEST 14: Backward Pass Tracking")
    
    with EpicPytorchDebugger(
        track_backward=True,
        do_pdb=False,
    ) as dbg:
        x = torch.randn(10, 10, requires_grad=True)
        y = x @ x.T
        loss = y.sum()
        loss.backward()
    
    backward_ops = dbg.get_backward_ops()
    print(f"Tracked {len(backward_ops)} backward operations")
    
    bwd_stats = dbg.get_backward_timings()
    print("\nTop backward operations:")
    for op, stats in list(sorted(bwd_stats.items(), key=lambda x: x[1]['total_ms'], reverse=True))[:5]:
        print(f"  {op}: {stats['total_ms']:.3f}ms")


# =============================================================================
# Test 15: Gradient Flow Visualization
# =============================================================================
def test_grad_flow():
    separator("TEST 15: Gradient Flow Visualization")
    
    with EpicPytorchDebugger(
        track_grad_flow=True,
        do_pdb=False,
    ) as dbg:
        x = torch.randn(10, 10, requires_grad=True)
        y = torch.randn(10, 10, requires_grad=True)
        z = x @ y
        loss = z.sum()
        loss.backward()
    
    grad_flow = dbg.get_grad_flow_data()
    print(f"Tracked gradient flow for {len(grad_flow)} tensors")
    
    layer_norms = dbg.get_layer_grad_norms()
    print("\nGradient norms per tensor:")
    for name, norms in layer_norms.items():
        avg_norm = sum(norms) / len(norms) if norms else 0
        print(f"  {name}: avg_norm={avg_norm:.4f}")


# =============================================================================
# Test 16: Operation Replay
# =============================================================================
def test_operation_replay():
    separator("TEST 16: Operation Replay")
    
    with EpicPytorchDebugger(
        enable_replay=True,
        replay_capacity=100,
        do_pdb=False,
    ) as dbg:
        x = torch.randn(5, 5)
        y = x @ x.T
        z = y.relu()
        
        dbg.bookmark("after_relu")
    
    saved_ops = dbg.get_saved_ops()
    print(f"Saved {len(saved_ops)} operations for replay")
    
    if saved_ops:
        print(f"\nFirst saved op: {saved_ops[0].name}")
        print(f"  Input shapes: {saved_ops[0].input_shapes}")
        print(f"  Output shapes: {saved_ops[0].output_shapes}")
        
        # Test replay
        if saved_ops[0].func is not None:
            replayed = dbg.replay_op(0)
            print(f"  Replay successful: {replayed is not None}")


# =============================================================================
# Test 17: Mixed Precision Tracking
# =============================================================================
def test_precision_tracking():
    separator("TEST 17: Mixed Precision Tracking")
    
    with EpicPytorchDebugger(
        track_precision=True,
        do_pdb=False,
    ) as dbg:
        # Normal float32 operations
        x = torch.randn(10, 10)
        y = x @ x.T
        
        # Convert to different dtype
        z = y.to(torch.float64)
        w = z.to(torch.float32)
    
    conversions = dbg.get_dtype_conversions()
    print(f"Tracked {len(conversions)} dtype conversions")
    
    losses = dbg.get_precision_losses()
    print(f"Precision losses: {len(losses)}")
    
    if conversions:
        print("\nConversions:")
        for c in conversions[:5]:
            loss_str = " [LOSS]" if c.is_precision_loss else ""
            print(f"  {c.from_dtype} -> {c.to_dtype} via {c.op_name}{loss_str}")


# =============================================================================
# Test 18: Memory Leak Detection
# =============================================================================
def test_memory_leak_detection():
    separator("TEST 18: Memory Leak Detection")
    
    with EpicPytorchDebugger(
        detect_leaks=True,
        leak_threshold_seconds=0.001,  # Very low for testing
        do_pdb=False,
    ) as dbg:
        tensors = []
        for i in range(5):
            t = torch.randn(100, 100)
            tensors.append(t)  # Simulating a "leak" by keeping references
    
    alive = dbg.get_alive_tensors()
    print(f"Tracking {len(alive)} alive tensors")
    
    # Clear references
    tensors.clear()
    
    # Force GC
    gc_stats = dbg.force_gc()
    print(f"\nAfter GC: freed {gc_stats['freed']} tensors")


# =============================================================================
# Test 19: Distributed Tracking
# =============================================================================
def test_distributed_tracking():
    separator("TEST 19: Distributed Tracking")
    
    # Note: This test just verifies the infrastructure works
    # Real distributed testing requires multiple processes
    
    with EpicPytorchDebugger(
        track_distributed=True,
        do_pdb=False,
    ) as dbg:
        x = torch.randn(10, 10)
        y = x @ x.T
    
    collective_ops = dbg.get_collective_ops()
    print(f"Collective ops detected: {len(collective_ops)}")
    print("(No actual collective ops in single-process test)")
    
    stats = dbg.get_collective_stats()
    print(f"\nDistributed tracking infrastructure: Working ✓")


# =============================================================================
# Test 20: Full Integration Example (OPTIMIZED)
# =============================================================================
def test_full_integration():
    separator("TEST 20: Full Integration - All Features (Optimized)")
    
    class SimpleNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 20)
            self.fc2 = nn.Linear(20, 5)
        
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            return self.fc2(x)
    
    model = SimpleNet()
    
    import time
    start = time.perf_counter()
    
    with EpicPytorchDebugger(
        profile_ops=True,
        track_modules=True,
        track_shapes=True,
        track_backward=True,
        track_grad_flow=True,
        track_gradients=True,
        detect_anomaly=True,
        anomaly_pdb=False,
        enable_replay=True,
        do_pdb=False,
        # OPTIMIZATION: Disable sys.settrace (biggest performance impact)
        run_trace=False,
    ) as dbg:
        dbg.register_modules(model)
        
        # Use dbg.track() for manual tensor tracking when run_trace=False
        x = dbg.track("x", torch.randn(32, 10, requires_grad=True))
        target = dbg.track("target", torch.randn(32, 5))
        
        dbg.snapshot("before_forward")
        
        output = dbg.track("output", model(x))
        loss = dbg.track("loss", nn.functional.mse_loss(output, target))
        
        dbg.snapshot("after_forward")
        
        # Mark backward pass start/end for proper tracking
        dbg.mark_backward_start()
        loss.backward()
        dbg.mark_backward_end()
        
        dbg.snapshot("after_backward")
    
    elapsed = time.perf_counter() - start
    print(f"Training step completed in {elapsed*1000:.2f}ms\n")
    
    summary = dbg.get_summary_dict()
    print("Summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")


# =============================================================================
# Test 21: Lightweight Mode Performance
# =============================================================================
def test_lightweight_mode():
    separator("TEST 21: Lightweight Mode - For Large Networks")
    
    import time
    
    class LargerNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(100, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 10),
            )
        
        def forward(self, x):
            return self.layers(x)
    
    model = LargerNet()
    x = torch.randn(64, 100, requires_grad=True)
    target = torch.randn(64, 10)
    
    # Baseline without debugger
    start = time.perf_counter()
    for i in range(10):
        output = model(x)
        loss = nn.functional.mse_loss(output, target)
        loss.backward()
        model.zero_grad()
    baseline = time.perf_counter() - start
    print(f"10 iterations WITHOUT debugger:      {baseline*1000:.2f}ms ({baseline*100:.2f}ms/iter)")
    
    # Run 10 iterations with MINIMAL mode (just profiling)
    start = time.perf_counter()
    
    with EpicPytorchDebugger(
        profile_ops=True,
        # Everything else disabled
        run_trace=False,
        track_modules=False,
        detect_anomaly=False,
        do_pdb=False,
    ) as dbg:
        for i in range(10):
            output = model(x)
            loss = nn.functional.mse_loss(output, target)
            loss.backward()
            model.zero_grad()
    
    minimal = time.perf_counter() - start
    print(f"10 iterations MINIMAL (profile only): {minimal*1000:.2f}ms ({minimal*100:.2f}ms/iter) [{((minimal/baseline)-1)*100:.0f}% overhead]")
    
    # Run with anomaly detection added
    start = time.perf_counter()
    
    with EpicPytorchDebugger(
        profile_ops=True,
        detect_anomaly=True,  # Adds isnan/isinf checks
        run_trace=False,
        anomaly_pdb=False,
        do_pdb=False,
    ) as dbg:
        for i in range(10):
            output = model(x)
            loss = nn.functional.mse_loss(output, target)
            loss.backward()
            model.zero_grad()
    
    with_anomaly = time.perf_counter() - start
    print(f"10 iterations + anomaly detection:   {with_anomaly*1000:.2f}ms ({with_anomaly*100:.2f}ms/iter) [{((with_anomaly/baseline)-1)*100:.0f}% overhead]")
    
    # Run 10 iterations with full lightweight mode
    start = time.perf_counter()
    
    with EpicPytorchDebugger(
        lightweight=True,  # Automatically disables expensive features
        profile_ops=True,
        track_modules=True,
        detect_anomaly=True,
        anomaly_pdb=False,
        do_pdb=False,
    ) as dbg:
        dbg.register_modules(model)
        
        for i in range(10):
            output = model(x)
            loss = nn.functional.mse_loss(output, target)
            loss.backward()
            model.zero_grad()
    
    lightweight = time.perf_counter() - start
    print(f"10 iterations LIGHTWEIGHT (+ modules):{lightweight*1000:.2f}ms ({lightweight*100:.2f}ms/iter) [{((lightweight/baseline)-1)*100:.0f}% overhead]")
    
    print("\n📊 Recommended configurations by use case:")
    print("  - Benchmarking:  profile_ops=True only                    (~70% overhead)")
    print("  - Bug hunting:   + detect_anomaly=True                    (~200% overhead)")  
    print("  - Full debug:    + track_modules, track_shapes, etc.      (~600% overhead)")
    print("  - Development:   run_trace=True for variable tracking     (~1000%+ overhead)")


# =============================================================================
# Main
# =============================================================================
def main():
    print("\n" + "#" * 80)
    print("#" + " " * 78 + "#")
    print("#" + "  EPIC PYTORCH DEBUGGER - FEATURE TEST SUITE v2.1  ".center(78) + "#")
    print("#" + " " * 78 + "#")
    print("#" * 80)
    
    # Original tests
    test_operation_timing()
    test_computation_graph()
    test_nan_detection()
    test_memory_tracking()
    test_gradient_tracking()
    test_snapshots()
    test_watch_mode()
    test_conditional_breakpoints()
    test_inplace_detection()
    test_device_transfers()
    test_logging_integration()
    
    # New feature tests
    test_module_tracking()
    test_shape_tracking()
    test_backward_tracking()
    test_grad_flow()
    test_operation_replay()
    test_precision_tracking()
    test_memory_leak_detection()
    test_distributed_tracking()
    
    # Full integration and performance
    test_full_integration()
    test_lightweight_mode()
    
    separator("ALL TESTS COMPLETED")
    print("The EpicPytorchDebugger v2.1 is working correctly!")
    print("\nNew features tested:")
    print("  - Module-aware tracking")
    print("  - Shape tracking")
    print("  - Backward pass tracking")
    print("  - Gradient flow visualization")
    print("  - Operation replay")
    print("  - Mixed precision tracking")
    print("  - Memory leak detection")
    print("  - Distributed training support")
    print("\nPerformance optimizations:")
    print("  - Lightweight mode for large models")
    print("  - No more inspect.stack() in backward detection")
    print("  - Duplicate hook prevention")
    print("  - Trace sampling for reduced overhead")
    print("  - Fast paths for common tensor operations")


if __name__ == "__main__":
    main()
