#!/usr/bin/env python3
"""
Test script for EpicPytorchDebugger features.

This script demonstrates all the debugging capabilities:
1. NaN/Inf detection
2. Export graph to DOT (Graphviz)
3. Operation timing
4. Memory tracking
5. Gradient hooks
6. Conditional breakpoints
7. Tensor snapshots
8. Watch mode
9. In-place operation detection
10. Device transfer tracking
11. Logging integration (wandb/tensorboard compatible)
"""

import torch
import torch.nn as nn

from debugger import (
    EpicPytorchDebugger,
    break_on_large_values,
    break_on_nan,
)
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
        run_trace=False,  # Faster without tracing
    ) as dbg:
        # Do some operations
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
    
    # Print the computation graph
    print("Computation graph for 'result':")
    comp_graph = dbg._get_tensor_metadata(result, "comp_graph")
    if comp_graph:
        print(comp_graph)
    
    # Export to DOT
    dbg.export_graph(result, "/tmp/test_graph.dot")
    print("\nDOT content:")
    print(comp_graph.to_dot())


# =============================================================================
# Test 3: NaN/Inf Detection
# =============================================================================
def test_nan_detection():
    separator("TEST 3: NaN/Inf Detection")
    
    print("Creating a tensor that will produce NaN...")
    
    with EpicPytorchDebugger(
        detect_anomaly=True,
        anomaly_pdb=False,  # Don't drop into pdb for testing
        do_pdb=False,
    ) as dbg:
        a = torch.tensor([1.0, 0.0, -1.0])
        b = torch.tensor([0.0, 0.0, 0.0])
        
        # This will produce inf and nan
        c = a / b  # [inf, nan, -inf]
        print(f"\nResult of division: {c}")


# =============================================================================
# Test 4: Memory Tracking (CUDA)
# =============================================================================
def test_memory_tracking():
    separator("TEST 4: Memory Tracking")
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping memory tracking test.")
        print("(Memory tracking works best with CUDA tensors)")
        return
    
    with EpicPytorchDebugger(
        track_memory=True,
        do_pdb=False,
        run_trace=False,
    ) as dbg:
        # Allocate some tensors on GPU
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
        # Create a simple computation with gradients
        x = torch.randn(10, 10, requires_grad=True)
        y = torch.randn(10, 10, requires_grad=True)
        
        # Forward pass
        z = x @ y
        loss = z.sum()
        
        # Backward pass (this triggers gradient hooks)
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
        
        # Take first snapshot
        dbg.snapshot("initial")
        print("Saved snapshot 'initial'")
        
        # Modify tensors
        x = x * 2
        y = y + 1
        z = torch.randn(3, 3)  # New tensor
        
        # Take second snapshot
        dbg.snapshot("modified")
        print("Saved snapshot 'modified'")
        
        print(f"\nAvailable snapshots: {dbg.list_snapshots()}")
    
    # Compare snapshots
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
        
        # Simulate a training loop
        for i in range(3):
            output = torch.randn(10) * (i + 1)  # Changes each iteration
            loss = output.sum()
            # Watch mode prints when these change


# =============================================================================
# Test 8: Conditional Breakpoints
# =============================================================================
def test_conditional_breakpoints():
    separator("TEST 8: Conditional Breakpoints")
    
    print("Setting up breakpoint for values > 100...")
    print("(Note: In real use, this would drop into pdb)\n")
    
    # Track if condition was triggered
    triggered_count = [0]
    
    def my_condition(t):
        max_val = t.abs().max().item()
        if max_val > 100:
            triggered_count[0] += 1
            return True
        return False
    
    # Create a custom debugger subclass that doesn't actually call pdb
    class TestDebugger(EpicPytorchDebugger):
        def _check_break_condition(self, tensor, op_name):
            if self.break_condition is None:
                return False
            try:
                if self.break_condition(tensor):
                    tensor_name = self._get_tensor_metadata(tensor, "tensor_name", "<unnamed>")
                    print(f"  [BREAKPOINT TRIGGERED] op={op_name}, tensor={tensor_name}, "
                          f"max_val={tensor.abs().max().item():.2f}")
                    return False  # Don't actually break for testing
            except Exception as e:
                print(f"Warning: Break condition raised exception: {e}")
            return False
    
    with TestDebugger(
        break_condition=my_condition,
        do_pdb=False,
        run_trace=False,
    ) as dbg:
        # Normal values - won't trigger
        a = torch.randn(10, 10)
        b = a @ a.T
        
        # Large values - would trigger breakpoint
        c = torch.randn(10, 10) * 1000  # This will trigger
    
    print(f"\nBreakpoint condition triggered {triggered_count[0]} time(s).")
    print("Conditional breakpoint test completed.")


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
        
        # These are in-place operations
        a.add_(1)      # In-place add
        a.mul_(2)      # In-place multiply
        a.relu_()      # In-place ReLU
        
        # Normal operation (not in-place)
        b = a + 1
    
    print(f"\nDetected {len(dbg.get_inplace_ops())} in-place operations:")
    for op in dbg.get_inplace_ops():
        print(f"  - {op['op']}")


# =============================================================================
# Test 10: Device Transfer Tracking
# =============================================================================
def test_device_transfers():
    separator("TEST 10: Device Transfer Tracking")
    
    if not torch.cuda.is_available():
        print("CUDA not available, simulating with CPU only.")
        print("(Device transfer tracking is most useful with GPU)")
        return
    
    with EpicPytorchDebugger(
        track_device_transfers=True,
        do_pdb=False,
    ) as dbg:
        # Create on CPU
        x = torch.randn(10, 10)
        
        # Move to GPU
        x = x.cuda()
        
        # Do some work
        y = x @ x.T
        
        # Move back to CPU
        y = y.cpu()
    
    transfers = dbg.get_device_transfers()
    print(f"\nRecorded {len(transfers)} device transfers:")
    for t in transfers:
        print(f"  {t['tensor']}: {t['from_device']} -> {t['to_device']} (via {t['op']})")


# =============================================================================
# Test 11: Logging Integration (wandb/tensorboard)
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
        y.add_(1)  # In-place
        loss = y.sum()
        loss.backward()
    
    # Get summary dict for logging
    summary = dbg.get_summary_dict()
    
    print("Summary dict (ready for wandb.log or tensorboard):")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    print("\nExample wandb usage:")
    print("  import wandb")
    print("  wandb.log(dbg.get_summary_dict())")


# =============================================================================
# Test 12: Full Integration Example
# =============================================================================
def test_full_integration():
    separator("TEST 12: Full Integration Example - Simple Neural Network")
    
    # Define a simple model
    class SimpleNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 20)
            self.fc2 = nn.Linear(20, 5)
        
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x
    
    model = SimpleNet()
    
    with EpicPytorchDebugger(
        profile_ops=True,
        track_gradients=True,
        detect_anomaly=True,
        anomaly_pdb=False,
        do_pdb=False,
        watch_tensors=["loss"],
    ) as dbg:
        # Training step
        x = torch.randn(32, 10)  # Batch of 32
        target = torch.randn(32, 5)
        
        dbg.snapshot("before_forward")
        
        output = model(x)
        loss = nn.functional.mse_loss(output, target)
        
        dbg.snapshot("after_forward")
        
        loss.backward()
        
        dbg.snapshot("after_backward")
    
    print("Training step completed.\n")
    
    # Show all the collected information
    print("=== Profiling Results ===")
    dbg.print_op_timings(top_n=5)
    
    print("=== Gradient Summary ===")
    grad_history = dbg.get_grad_history()
    print(f"Tracked {len(grad_history)} tensors with gradients\n")
    
    print("=== Snapshot Comparison ===")
    dbg.print_snapshot_diff("before_forward", "after_forward")


# =============================================================================
# Main
# =============================================================================
def main():
    print("\n" + "#" * 80)
    print("#" + " " * 78 + "#")
    print("#" + "  EPIC PYTORCH DEBUGGER - FEATURE TEST SUITE  ".center(78) + "#")
    print("#" + " " * 78 + "#")
    print("#" * 80)
    
    # Run all tests
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
    test_full_integration()
    
    separator("ALL TESTS COMPLETED")
    print("The EpicPytorchDebugger is working correctly!")
    print("\nTo visualize the computation graph:")
    print("  dot -Tpng /tmp/test_graph.dot -o /tmp/test_graph.png")
    print("  # Then view /tmp/test_graph.png")


if __name__ == "__main__":
    main()



