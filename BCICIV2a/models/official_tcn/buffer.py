
"""Utilities for passing streaming buffers through causal TCN layers."""

import torch

from typing import Optional
from typing import Union
from typing import List
from collections.abc import Iterable


class BufferIO():
    """Track input, output, and internally created buffers during streaming inference."""

    def __init__(
            self,
            in_buffers: Optional[ Iterable ] = None,
            ):
        if in_buffers is not None:
            self.in_buffers_length = len( in_buffers )
            self.in_buffers = iter( in_buffers )
        else:
            self.in_buffers_length = None
            self.in_buffers = None
        
        # out_buffers stores the buffers produced in the current step, while
        # internal_buffers captures layers that fell back to their own state.
        self.out_buffers = []
        self.internal_buffers = []
        return
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.in_buffers is not None:
            return next( self.in_buffers )
        else:
            return None
        
    def append_out_buffer(
            self,
            x: torch.Tensor,
            ):
        self.out_buffers.append(x)
        return
    
    def append_internal_buffer(
            self,
            x: torch.Tensor,
            ):
        self.internal_buffers.append(x)
        return
        
    def next_in_buffer(
            self,
            ):
        return self.__next__()
        
    def step(self):
        """Advance one streaming step by feeding produced buffers back in."""
        # If in_buffers is None, then the internal buffers are used as input
        # After the first step, the operation will continue as usual
        if self.in_buffers is None:
            self.in_buffers_length = len( self.internal_buffers)
        if len( self.out_buffers ) != self.in_buffers_length:
            raise ValueError(
                """
                Number of out buffers does not match number of in buffers.
                """
                )
        self.in_buffers = iter( self.out_buffers )
        self.out_buffers = []
        return
