"""
Debugging utilities for visualizing TPCF stitching results.
"""

import numpy as np


def print_stitch_summary(results):
    """
    Print a summary of the stitching results.
    
    Parameters
    ----------
    results : dict
        Output from get_joint_tpcf with return_debug_info=True
    """
    print("\n" + "="*60)
    print("TPCF STITCHING SUMMARY")
    print("="*60)
    
    # Transition points
    print("\nTransition Points (r_min, r_max):")
    print("-" * 40)
    for box_size in sorted(results['transition_points'].keys(), reverse=True):
        r_min, r_max = results['transition_points'][box_size]
        r_min_str = f"{r_min:.4f}" if r_min > 0 else "0.0"
        r_max_str = f"{r_max:.4f}" if r_max != np.inf else "inf"
        print(f"  L = {box_size:4d}: ({r_min_str:>10s}, {r_max_str:>10s})")
    
    # Points per box
    print("\nPoints Used Per Box:")
    print("-" * 40)
    total_points = len(results['radii'])
    for box_size in sorted(results['n_points_per_box'].keys(), reverse=True):
        n_points = results['n_points_per_box'][box_size]
        percentage = 100 * n_points / total_points
        print(f"  L = {box_size:4d}: {n_points:4d} points ({percentage:5.1f}%)")
    
    print(f"\n  Total: {total_points} points")
    
    # Radii range for each box
    print("\nRadii Range Per Box (in stitched data):")
    print("-" * 40)
    for box_size in sorted(results['n_points_per_box'].keys(), reverse=True):
        mask = results['box_labels'] == box_size
        if np.any(mask):
            r_box = results['radii'][mask]
            print(f"  L = {box_size:4d}: r = [{r_box.min():.4f}, {r_box.max():.4f}]")
    
    print("="*60 + "\n")


def get_box_color_map():
    """
    Get a consistent color map for different box sizes.
    
    Returns
    -------
    dict
        Dictionary mapping box_size -> color
    """
    return {
        32: 'C0',    # blue
        128: 'C1',   # orange
        512: 'C2',   # green
        2048: 'C3',  # red
    }


def plot_stitched_with_labels(results, ax=None, **plot_kwargs):
    """
    Plot stitched TPCF with colors indicating which box each point came from.
    
    Parameters
    ----------
    results : dict
        Output from get_joint_tpcf with return_debug_info=True
    ax : matplotlib axis, optional
        Axis to plot on. If None, uses current axis.
    **plot_kwargs
        Additional keyword arguments passed to errorbar
        
    Returns
    -------
    matplotlib axis
    """
    import matplotlib.pyplot as plt
    
    if ax is None:
        ax = plt.gca()
    
    color_map = get_box_color_map()
    
    # Plot each segment with appropriate color
    for box_size in sorted(np.unique(results['box_labels'])):
        mask = results['box_labels'] == box_size
        
        ax.errorbar(
            results['radii'][mask],
            results['xi'][mask],
            yerr=results['xi_err'][mask],
            fmt='.',
            color=color_map.get(box_size, 'gray'),
            label=f'$L = {box_size}$',
            **plot_kwargs
        )
    
    return ax


def check_transition_quality(sample_tpcfs, results, verbose=True):
    """
    Check the quality of transitions between boxes.
    
    Parameters
    ----------
    sample_tpcfs : dict
        Input dictionary passed to get_joint_tpcf
    results : dict
        Output from get_joint_tpcf with return_debug_info=True
    verbose : bool, optional
        If True, print detailed information
        
    Returns
    -------
    dict
        Dictionary with transition quality metrics
    """
    box_sizes = sorted(sample_tpcfs.keys(), reverse=True)
    transition_quality = {}

    if verbose:
        print("\n" + "="*60)
        print("TRANSITION QUALITY CHECK")
        print("="*60)

    for i in range(len(box_sizes) - 1):
        large_box = box_sizes[i]
        small_box = box_sizes[i + 1]

        transition_r = results['transition_points'][large_box][0]

        # Get xi values near transition from both boxes
        r_large = sample_tpcfs[large_box]['radii']
        xi_large = sample_tpcfs[large_box]['xi']
        r_small = sample_tpcfs[small_box]['radii']
        xi_small = sample_tpcfs[small_box]['xi']

        # Find closest points to transition
        idx_large = np.argmin(np.abs(r_large - transition_r))
        idx_small = np.argmin(np.abs(r_small - transition_r))

        # Compare values
        xi_large_at_trans = xi_large[idx_large]
        xi_small_at_trans = xi_small[idx_small]
        ratio = xi_large_at_trans / xi_small_at_trans

        transition_quality[f'{large_box}->{small_box}'] = {
            'transition_r': transition_r,
            'xi_large': xi_large_at_trans,
            'xi_small': xi_small_at_trans,
            'ratio': ratio,
        }

        if verbose:
            print(f"\nTransition: L={large_box} -> L={small_box}")
            print(f"  r = {transition_r:.4f}")
            print(f"  xi_large = {xi_large_at_trans:.4f}")
            print(f"  xi_small = {xi_small_at_trans:.4f}")
            print(f"  ratio = {ratio:.4f}")
            if ratio < 0.95:
                print(f"  ⚠ Large box underestimating by {(1-ratio)*100:.1f}%")
            elif ratio > 1.05:
                print(f"  ⚠ Large box overestimating by {(ratio-1)*100:.1f}%")
            else:
                print("  ✓ Good agreement")

    if verbose:
        print("="*60 + "\n")

    return transition_quality
