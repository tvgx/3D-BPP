from src.evaluation.packing_simulator import Bin

class FitnessEvaluator:
    """
    Evaluates the quality of a packing solution.
    """
    @staticmethod
    def calculate_anb(bins, bin_capacity=None):
        """
        Calculates Adjusted Number of Bins (aNB).
        Formula: aNB = N + (V_last_bin / V_total)
        
        Where:
        - N: Number of used bins
        - V_last_bin: Used volume of the 'last' bin (typically the least loaded or the fragmentation bin)
        - V_total: Total capacity of one bin (assuming homogeneous bins)
        """
        if not bins:
            return float('inf')
            
        num_bins = len(bins)
        
        # Identify the 'last' bin. 
        # In BPP optimization, the 'last' bin is often considered the one we want to eliminate,
        # which corresponds to the bin with the minimum load.
        # Let's find the bin with the minimum utilized volume.
        
        loads_vol = []
        capacity = 0
        for b in bins:
            used_vol = sum([item.volume for item, _, _ in b.items])
            loads_vol.append(used_vol)
            if capacity == 0:
                capacity = b.capacity
        
        # If user passed explicit bin_capacity, use it
        if bin_capacity:
            capacity = bin_capacity
            
        if capacity == 0: # Should not happen if bins exist
            return float('inf')

        v_last_bin = min(loads_vol) if loads_vol else 0
        v_total = capacity
        
        return num_bins + (v_last_bin / v_total)
