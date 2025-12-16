"""
Anomaly detection mixin - NaN/Inf detection and handling.
"""
import pdb
from typing import Any, Callable, Dict, Optional

import torch


class AnomalyDetectionMixin:
    """Mixin providing NaN/Inf detection capabilities."""
    
    # These will be set by the main debugger class
    detect_anomaly: bool
    anomaly_pdb: bool
    break_condition: Optional[Callable[[torch.Tensor], bool]]
    _get_tensor_metadata: Callable
    
    def _init_anomaly_detection(self) -> None:
        """Initialize anomaly detection state."""
        self._anomaly_count = 0
        self._anomaly_log = []
    
    def _check_anomaly(self, tensor: torch.Tensor, op_name: str) -> bool:
        """Check tensor for NaN/Inf values. Returns True if anomaly found."""
        if not self.detect_anomaly:
            return False
        
        try:
            has_nan = torch.isnan(tensor).any().item()
            has_inf = torch.isinf(tensor).any().item()
            
            if has_nan or has_inf:
                tensor_name = self._get_tensor_metadata(tensor, "tensor_name", "<unnamed>")
                anomaly_type = []
                if has_nan:
                    anomaly_type.append("NaN")
                if has_inf:
                    anomaly_type.append("Inf")
                
                self._anomaly_count += 1
                anomaly_info = {
                    'type': anomaly_type,
                    'op': op_name,
                    'tensor': tensor_name,
                    'shape': tuple(tensor.shape),
                    'device': str(tensor.device),
                }
                self._anomaly_log.append(anomaly_info)
                
                print("\n" + "!" * 60)
                print(f"ANOMALY DETECTED: {', '.join(anomaly_type)}")
                print(f"  Operation: {op_name}")
                print(f"  Tensor: {tensor_name}")
                print(f"  Shape: {tuple(tensor.shape)}")
                print(f"  Device: {tensor.device}")
                
                # Show computation graph if available
                comp_graph = self._get_tensor_metadata(tensor, "comp_graph")
                if comp_graph:
                    print(f"  Computation graph:\n{comp_graph}")
                
                print("!" * 60 + "\n")
                
                return True
        except Exception:
            pass
        
        return False
    
    def _check_break_condition(self, tensor: torch.Tensor, op_name: str) -> bool:
        """Check if break condition is met for a tensor."""
        if self.break_condition is None:
            return False
        
        try:
            if self.break_condition(tensor):
                tensor_name = self._get_tensor_metadata(tensor, "tensor_name", "<unnamed>")
                print("\n" + "=" * 60)
                print("BREAKPOINT CONDITION MET")
                print(f"  Operation: {op_name}")
                print(f"  Tensor: {tensor_name}")
                print(f"  Shape: {tuple(tensor.shape)}")
                print(f"  Device: {tensor.device}")
                print("=" * 60 + "\n")
                return True
        except Exception as e:
            print(f"Warning: Break condition raised exception: {e}")
        
        return False
    
    def _handle_anomaly(self) -> None:
        """Handle detected anomaly (drop into pdb if configured)."""
        if self.anomaly_pdb:
            print("Dropping into pdb due to anomaly...")
            pdb.set_trace()
    
    def get_anomaly_log(self) -> list:
        """Get the log of detected anomalies."""
        return self._anomaly_log
    
    def get_anomaly_count(self) -> int:
        """Get total count of detected anomalies."""
        return self._anomaly_count
