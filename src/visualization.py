import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

def create_color_map(bins):
    """
    Create a consistent color mapping for all items based on their IDs.
    Items with the same ID will always get the same color.
    
    Args:
        bins: List of Bin objects containing all items
        
    Returns:
        Dictionary mapping item_id to color string
    """
    # Collect all unique item IDs
    all_item_ids = set()
    for bin_obj in bins:
        for item, _, _ in bin_obj.items:
            all_item_ids.add(item.id)
    
    # Create extended color palette (use multiple colormaps for more colors)
    colors_list = []
    # Use tab20 (20 colors)
    colors_list.extend(plt.cm.tab20(np.linspace(0, 1, 20)))
    # Use Set3 (12 colors)
    colors_list.extend(plt.cm.Set3(np.linspace(0, 1, 12)))
    # Use Pastel1 (9 colors)
    colors_list.extend(plt.cm.Pastel1(np.linspace(0, 1, 9)))
    # Use Dark2 (8 colors)
    colors_list.extend(plt.cm.Dark2(np.linspace(0, 1, 8)))
    
    # Convert to RGB strings
    rgb_colors = [f'rgb({int(r*255)},{int(g*255)},{int(b*255)})' for r, g, b, _ in colors_list]
    
    # Create mapping: item_id -> color
    color_map = {}
    sorted_ids = sorted(all_item_ids)
    for idx, item_id in enumerate(sorted_ids):
        color_map[item_id] = rgb_colors[idx % len(rgb_colors)]
    
    return color_map

def plot_convergence(history, algorithm_name, save_path=None):
    """
    Plot convergence curve showing fitness over generations.
    
    Args:
        history: List of fitness values (best fitness per generation)
        algorithm_name: Name of the algorithm (for labeling)
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history, linewidth=2, label=f'{algorithm_name.upper()}')
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Best Fitness', fontsize=12)
    plt.title(f'Convergence Curve - {algorithm_name.upper()}', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Convergence plot saved to: {save_path}")
    else:
        plt.savefig('convergence_plot.png', dpi=300, bbox_inches='tight')
        print("Convergence plot saved to: convergence_plot.png")
    
    plt.close()

def plot_convergence_comparison(histories_dict, save_path=None):
    """
    Plot convergence curves for multiple algorithms for comparison.
    
    Args:
        histories_dict: Dictionary with algorithm names as keys and history lists as values
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(12, 7))
    
    for algo_name, history in histories_dict.items():
        plt.plot(history, linewidth=2, label=f'{algo_name.upper()}', marker='o', markersize=3, markevery=max(1, len(history)//20))
    
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Best Fitness', fontsize=12)
    plt.title('Algorithm Comparison - Convergence Curves', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to: {save_path}")
    else:
        plt.savefig('convergence_comparison.png', dpi=300, bbox_inches='tight')
        print("Comparison plot saved to: convergence_comparison.png")
    
    plt.close()

def visualize_3d_packing(bins, bin_dims, algorithm_name="", save_path=None):
    """
    Create interactive 3D visualization of bin packing using Plotly.
    
    Args:
        bins: List of Bin objects containing packed items
        bin_dims: Tuple (D, W, H) of bin dimensions
        algorithm_name: Name of algorithm (for title)
        save_path: Optional path to save HTML file
    """
    if not bins:
        print("No bins to visualize!")
        return
    
    # Create subplots - one for each bin
    n_bins = len(bins)
    cols = min(3, n_bins)  # Max 3 columns
    rows = (n_bins + cols - 1) // cols
    
    fig = make_subplots(
        rows=rows, cols=cols,
        specs=[[{'type': 'scatter3d'} for _ in range(cols)] for _ in range(rows)],
        subplot_titles=[f'Bin {i+1}' for i in range(n_bins)],
        horizontal_spacing=0.1,
        vertical_spacing=0.15
    )
    
    # Create consistent color mapping for all items
    color_map = create_color_map(bins)
    
    for bin_idx, bin_obj in enumerate(bins):
        row = bin_idx // cols + 1
        col = bin_idx % cols + 1
        
        # Draw bin outline (wireframe)
        bin_d, bin_w, bin_h = bin_dims
        bin_corners = [
            [0, 0, 0], [bin_d, 0, 0], [bin_d, bin_w, 0], [0, bin_w, 0],
            [0, 0, bin_h], [bin_d, 0, bin_h], [bin_d, bin_w, bin_h], [0, bin_w, bin_h]
        ]
        
        # Bin edges (12 edges of a box)
        bin_edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
            [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
            [0, 4], [1, 5], [2, 6], [3, 7]   # Vertical edges
        ]
        
        # Draw bin wireframe
        for edge in bin_edges:
            x_edge = [bin_corners[edge[0]][0], bin_corners[edge[1]][0]]
            y_edge = [bin_corners[edge[0]][1], bin_corners[edge[1]][1]]
            z_edge = [bin_corners[edge[0]][2], bin_corners[edge[1]][2]]
            
            fig.add_trace(
                go.Scatter3d(
                    x=x_edge, y=y_edge, z=z_edge,
                    mode='lines',
                    line=dict(color='grey', width=2),
                    showlegend=False,
                    hoverinfo='skip'
                ),
                row=row, col=col
            )
        
        # Draw each item in the bin
        for item_idx, (item, pos, dims_rot) in enumerate(bin_obj.items):
            x, y, z = pos
            d, w, h = dims_rot
            
            # Create 8 corners of the item box
            item_corners = [
                [x, y, z], [x+d, y, z], [x+d, y+w, z], [x, y+w, z],
                [x, y, z+h], [x+d, y, z+h], [x+d, y+w, z+h], [x, y+w, z+h]
            ]
            
            # Create mesh3d for filled box
            i_indices = [7,0,0,0,4,4,6,6,4,0,3,2]
            j_indices = [3,4,1,2,5,6,5,2,0,1,6,3]
            k_indices = [0,7,2,3,6,7,1,1,5,5,7,6]
            
            x_coords = [item_corners[i][0] for i in range(8)]
            y_coords = [item_corners[i][1] for i in range(8)]
            z_coords = [item_corners[i][2] for i in range(8)]
            
            # Use consistent color mapping - same ID = same color
            color = color_map.get(item.id, 'rgb(128,128,128)')  # Default grey if ID not found
            
            fig.add_trace(
                go.Mesh3d(
                    x=x_coords, y=y_coords, z=z_coords,
                    i=i_indices, j=j_indices, k=k_indices,
                    color=color,
                    opacity=0.8,
                    flatshading=True,
                    name=f'Item {item.id}',
                    showlegend=(bin_idx == 0),  # Only show legend for first bin
                    hovertemplate=f'Item {item.id}<br>Position: ({x}, {y}, {z})<br>Size: {d}×{w}×{h}<br>Weight: {item.weight:.1f}<extra></extra>'
                ),
                row=row, col=col
            )
        
        # Update scene for this specific subplot (will be done after all traces are added)
    
    # Update scene settings for each subplot
    bin_d, bin_w, bin_h = bin_dims
    for bin_idx in range(n_bins):
        scene_name = f'scene{bin_idx+1}' if bin_idx > 0 else 'scene'
        fig.update_layout(**{
            scene_name: dict(
                xaxis=dict(title='Depth (D)', range=[0, bin_d]),
                yaxis=dict(title='Width (W)', range=[0, bin_w]),
                zaxis=dict(title='Height (H)', range=[0, bin_h]),
                aspectmode='cube',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            )
        })
    
    # Update overall layout
    title_text = f'3D Bin Packing Visualization'
    if algorithm_name:
        title_text += f' - {algorithm_name.upper()}'
    
    fig.update_layout(
        title=dict(
            text=title_text,
            x=0.5,
            font=dict(size=16, color='black')
        ),
        height=400 * rows,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Save or show
    if save_path:
        fig.write_html(save_path)
        print(f"3D visualization saved to: {save_path}")
    else:
        default_path = f'result_visualization_{algorithm_name.lower()}.html'
        fig.write_html(default_path)
        print(f"3D visualization saved to: {default_path}")
    
    return fig

def visualize_single_bin(bin_obj, bin_dims, bin_idx=0, algorithm_name="", save_path=None, color_map=None):
    """
    Create interactive 3D visualization for a single bin.
    
    Args:
        bin_obj: Bin object containing packed items
        bin_dims: Tuple (D, W, H) of bin dimensions
        bin_idx: Index of the bin (for labeling)
        algorithm_name: Name of algorithm (for title)
        save_path: Optional path to save HTML file
        color_map: Optional color mapping dictionary (for consistency across bins)
    """
    fig = go.Figure()
    
    # Draw bin outline (wireframe)
    bin_d, bin_w, bin_h = bin_dims
    bin_corners = [
        [0, 0, 0], [bin_d, 0, 0], [bin_d, bin_w, 0], [0, bin_w, 0],
        [0, 0, bin_h], [bin_d, 0, bin_h], [bin_d, bin_w, bin_h], [0, bin_w, bin_h]
    ]
    
    bin_edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7]
    ]
    
    # Draw bin wireframe
    for edge in bin_edges:
        x_edge = [bin_corners[edge[0]][0], bin_corners[edge[1]][0]]
        y_edge = [bin_corners[edge[0]][1], bin_corners[edge[1]][1]]
        z_edge = [bin_corners[edge[0]][2], bin_corners[edge[1]][2]]
        
        fig.add_trace(
            go.Scatter3d(
                x=x_edge, y=y_edge, z=z_edge,
                mode='lines',
                line=dict(color='grey', width=3),
                name='Bin Outline',
                showlegend=False,
                hoverinfo='skip'
            )
        )
    
    # Create consistent color mapping
    if color_map is None:
        # If no color map provided, create one from this bin's items
        temp_bins = [bin_obj]
        color_map = create_color_map(temp_bins)
    
    # Draw each item
    for item_idx, (item, pos, dims_rot) in enumerate(bin_obj.items):
        x, y, z = pos
        d, w, h = dims_rot
        
        item_corners = [
            [x, y, z], [x+d, y, z], [x+d, y+w, z], [x, y+w, z],
            [x, y, z+h], [x+d, y, z+h], [x+d, y+w, z+h], [x, y+w, z+h]
        ]
        
        i_indices = [7,0,0,0,4,4,6,6,4,0,3,2]
        j_indices = [3,4,1,2,5,6,5,2,0,1,6,3]
        k_indices = [0,7,2,3,6,7,1,1,5,5,7,6]
        
        x_coords = [item_corners[i][0] for i in range(8)]
        y_coords = [item_corners[i][1] for i in range(8)]
        z_coords = [item_corners[i][2] for i in range(8)]
        
        # Use consistent color mapping - same ID = same color
        color = color_map.get(item.id, 'rgb(128,128,128)')  # Default grey if ID not found
        
        fig.add_trace(
            go.Mesh3d(
                x=x_coords, y=y_coords, z=z_coords,
                i=i_indices, j=j_indices, k=k_indices,
                color=color,
                opacity=0.8,
                flatshading=True,
                name=f'Item {item.id}',
                hovertemplate=f'Item {item.id}<br>Position: ({x}, {y}, {z})<br>Size: {d}×{w}×{h}<br>Weight: {item.weight:.1f}<extra></extra>'
            )
        )
    
    title_text = f'Bin {bin_idx + 1} - 3D Packing Visualization'
    if algorithm_name:
        title_text += f' ({algorithm_name.upper()})'
    
    fig.update_layout(
        title=dict(
            text=title_text,
            x=0.5,
            font=dict(size=16, color='black')
        ),
        scene=dict(
            xaxis=dict(title='Depth (D)', range=[0, bin_d]),
            yaxis=dict(title='Width (W)', range=[0, bin_w]),
            zaxis=dict(title='Height (H)', range=[0, bin_h]),
            aspectmode='cube',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        height=800,
        showlegend=True
    )
    
    if save_path:
        fig.write_html(save_path)
        print(f"3D visualization saved to: {save_path}")
    else:
        default_path = f'result_visualization_bin_{bin_idx}_{algorithm_name.lower()}.html'
        fig.write_html(default_path)
        print(f"3D visualization saved to: {default_path}")
    
    return fig

