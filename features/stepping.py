"""
Interactive Step-Through mixin - Step through operations one at a time.
"""
import pdb
import sys
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import torch


class StepController:
    """Controller for interactive stepping through operations."""
    
    def __init__(self):
        self.stepping_enabled = False
        self.step_mode = 'run'  # 'run', 'step', 'next', 'continue'
        self.breakpoint_ops: Set[str] = set()
        self.breakpoint_modules: Set[str] = set()
        self.skip_ops: Set[str] = set()
        self.step_count = 0
        self.stop_at_step: Optional[int] = None
        self.paused = False
        self._last_op_info: Optional[Dict] = None
    
    def should_pause(self, op_name: str, module_name: Optional[str], step_num: int) -> bool:
        """Determine if we should pause at this operation."""
        if not self.stepping_enabled:
            return False
        
        # Skip specified ops
        if any(skip in op_name for skip in self.skip_ops):
            return False
        
        # Check specific step number
        if self.stop_at_step is not None and step_num >= self.stop_at_step:
            self.stop_at_step = None
            return True
        
        # Check breakpoint ops
        if any(bp in op_name for bp in self.breakpoint_ops):
            return True
        
        # Check breakpoint modules
        if module_name and any(bp in module_name for bp in self.breakpoint_modules):
            return True
        
        # Check step modes
        if self.step_mode == 'step':
            return True
        
        return False


class SteppingMixin:
    """Mixin providing interactive step-through debugging."""
    
    # Configuration
    enable_stepping: bool
    track_modules: bool
    _get_current_module: Callable
    
    def _init_stepping(self, enable_stepping: bool = False) -> None:
        """Initialize stepping state."""
        self.enable_stepping = enable_stepping
        self._step_controller = StepController()
        self._step_count = 0
        self._step_history: List[Dict[str, Any]] = []
    
    def _format_tensor_preview(self, tensor: torch.Tensor, max_elements: int = 6) -> str:
        """Format a tensor preview for display."""
        flat = tensor.flatten()
        if flat.numel() <= max_elements:
            values = flat.tolist()
        else:
            half = max_elements // 2
            values = flat[:half].tolist() + ['...'] + flat[-half:].tolist()
        return f"tensor({values}, shape={tuple(tensor.shape)}, dtype={tensor.dtype})"
    
    def _format_step_info(
        self,
        op_name: str,
        args: Tuple,
        kwargs: Dict,
        module_name: Optional[str],
    ) -> str:
        """Format operation info for stepping display."""
        lines = []
        lines.append(f"\n{'='*60}")
        lines.append(f"STEP {self._step_count}: {op_name}")
        if module_name:
            lines.append(f"Module: {module_name}")
        lines.append(f"{'='*60}")
        
        # Show input tensors
        lines.append("\nInputs:")
        for i, arg in enumerate(args):
            if isinstance(arg, torch.Tensor):
                lines.append(f"  arg[{i}]: {self._format_tensor_preview(arg)}")
        
        for key, val in kwargs.items():
            if isinstance(val, torch.Tensor):
                lines.append(f"  {key}: {self._format_tensor_preview(val)}")
        
        return "\n".join(lines)
    
    def _handle_step_pause(
        self,
        op_name: str,
        func: Callable,
        args: Tuple,
        kwargs: Dict,
        module_name: Optional[str],
    ) -> Optional[Any]:
        """
        Handle a pause during stepping. Shows interactive menu.
        Returns None to continue, or a value to return immediately.
        """
        if not self.enable_stepping:
            return None
        
        if not self._step_controller.should_pause(op_name, module_name, self._step_count):
            return None
        
        # Store current op info
        self._step_controller._last_op_info = {
            'op': op_name,
            'func': func,
            'args': args,
            'kwargs': kwargs,
            'module': module_name,
            'step': self._step_count,
        }
        
        # Print step info
        print(self._format_step_info(op_name, args, kwargs, module_name))
        
        # Interactive command loop
        while True:
            try:
                cmd = input("\n[s]tep | [n]ext N | [c]ontinue | [b]reak op | [i]nspect | [p]db | [r]un > ").strip().lower()
            except EOFError:
                cmd = 'r'
            
            if cmd == 's' or cmd == 'step' or cmd == '':
                self._step_controller.step_mode = 'step'
                break
            
            elif cmd.startswith('n') or cmd.startswith('next'):
                parts = cmd.split()
                n = int(parts[1]) if len(parts) > 1 else 1
                self._step_controller.stop_at_step = self._step_count + n
                self._step_controller.step_mode = 'run'
                break
            
            elif cmd == 'c' or cmd == 'continue':
                self._step_controller.step_mode = 'run'
                break
            
            elif cmd.startswith('b') or cmd.startswith('break'):
                parts = cmd.split(maxsplit=1)
                if len(parts) > 1:
                    bp = parts[1]
                    self._step_controller.breakpoint_ops.add(bp)
                    print(f"Added breakpoint on ops containing: {bp}")
                else:
                    print(f"Current breakpoint ops: {self._step_controller.breakpoint_ops}")
                    print(f"Current breakpoint modules: {self._step_controller.breakpoint_modules}")
            
            elif cmd == 'i' or cmd == 'inspect':
                self._inspect_step_args(args, kwargs)
            
            elif cmd == 'p' or cmd == 'pdb':
                print("Entering pdb. Type 'c' to continue stepping.")
                pdb.set_trace()
            
            elif cmd == 'r' or cmd == 'run':
                self._step_controller.stepping_enabled = False
                break
            
            elif cmd == 'h' or cmd == 'help':
                self._print_step_help()
            
            else:
                print(f"Unknown command: {cmd}. Type 'h' for help.")
        
        return None
    
    def _inspect_step_args(self, args: Tuple, kwargs: Dict) -> None:
        """Inspect arguments in detail."""
        print("\n--- Detailed Argument Inspection ---")
        for i, arg in enumerate(args):
            print(f"\nargs[{i}]:")
            if isinstance(arg, torch.Tensor):
                print(f"  shape: {tuple(arg.shape)}")
                print(f"  dtype: {arg.dtype}")
                print(f"  device: {arg.device}")
                print(f"  requires_grad: {arg.requires_grad}")
                if arg.numel() <= 100:
                    print(f"  values:\n{arg}")
                else:
                    print(f"  min: {arg.min().item():.6f}")
                    print(f"  max: {arg.max().item():.6f}")
                    print(f"  mean: {arg.float().mean().item():.6f}")
                    print(f"  has_nan: {torch.isnan(arg).any().item()}")
                    print(f"  has_inf: {torch.isinf(arg).any().item()}")
            else:
                print(f"  type: {type(arg)}")
                print(f"  value: {arg}")
        
        for key, val in kwargs.items():
            print(f"\nkwargs['{key}']:")
            if isinstance(val, torch.Tensor):
                print(f"  shape: {tuple(val.shape)}")
                print(f"  dtype: {val.dtype}")
            else:
                print(f"  type: {type(val)}, value: {val}")
    
    def _print_step_help(self) -> None:
        """Print stepping help."""
        print("""
Stepping Commands:
  s, step      - Execute this op and pause at the next one
  n, next [N]  - Execute N operations then pause (default: 1)
  c, continue  - Continue until a breakpoint
  b, break OP  - Add breakpoint on ops containing OP
  i, inspect   - Detailed inspection of current arguments
  p, pdb       - Drop into Python debugger
  r, run       - Disable stepping and run to completion
  h, help      - Show this help
        """)
    
    # Public API
    
    def start_stepping(self) -> None:
        """Enable stepping mode."""
        self.enable_stepping = True
        self._step_controller.stepping_enabled = True
        self._step_controller.step_mode = 'step'
        print("Stepping enabled. Will pause at each operation.")
    
    def stop_stepping(self) -> None:
        """Disable stepping mode."""
        self._step_controller.stepping_enabled = False
        self._step_controller.step_mode = 'run'
        print("Stepping disabled.")
    
    def add_op_breakpoint(self, op_pattern: str) -> None:
        """Add a breakpoint for operations matching pattern."""
        self._step_controller.breakpoint_ops.add(op_pattern)
    
    def add_module_breakpoint(self, module_pattern: str) -> None:
        """Add a breakpoint for modules matching pattern."""
        self._step_controller.breakpoint_modules.add(module_pattern)
    
    def clear_breakpoints(self) -> None:
        """Clear all breakpoints."""
        self._step_controller.breakpoint_ops.clear()
        self._step_controller.breakpoint_modules.clear()
    
    def skip_op(self, op_pattern: str) -> None:
        """Skip pausing on operations matching pattern."""
        self._step_controller.skip_ops.add(op_pattern)
    
    def get_step_history(self) -> List[Dict[str, Any]]:
        """Get history of stepped operations."""
        return self._step_history
