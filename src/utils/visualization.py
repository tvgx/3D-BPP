import plotly.graph_objects as go
import numpy as np
import random

class Visualizer:
    @staticmethod
    def _get_random_color():
        return f'rgb({random.randint(50, 200)}, {random.randint(50, 200)}, {random.randint(50, 200)})'

    @staticmethod
    def draw_box(fig, x, y, z, dx, dy, dz, color='blue', opacity=1.0, name="", show_legend=False):
        x_corners = [x, x+dx, x+dx, x, x, x+dx, x+dx, x]
        y_corners = [y, y, y+dy, y+dy, y, y, y+dy, y+dy]
        z_corners = [z, z, z, z, z+dz, z+dz, z+dz, z+dz]
        
        i = [7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2]
        j = [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3]
        k = [0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6]
        
        fig.add_trace(go.Mesh3d(
            x=x_corners, y=y_corners, z=z_corners,
            i=i, j=j, k=k,
            color=color, opacity=opacity, name=name, showlegend=show_legend, flatshading=True, hoverinfo='name'
        ))

    @staticmethod
    def visualize_solution(bins, save_path=None):
        figures = []
        for b_idx, b in enumerate(bins):
            fig = go.Figure()
            bd, bw, bh = b.dims
            Visualizer.draw_box(fig, 0, 0, 0, bd, bw, bh, color='grey', opacity=0.1, name=f"Bin {b.id}")
            
            for item_tuple in b.items:
                item_obj, pos, dims = item_tuple
                x, y, z = pos
                d, w, h = dims
                color = Visualizer._get_random_color()
                Visualizer.draw_box(fig, x, y, z, d, w, h, color=color, opacity=1.0, name=f"Item {item_obj.id}")
            
            fig.update_layout(title=f"Bin {b.id}", scene=dict(xaxis=dict(title='X'), yaxis=dict(title='Y'), zaxis=dict(title='Z')))
            figures.append(fig)
            if save_path: 
                p = save_path.replace(".html", f"_bin_{b.id}.html")
                fig.write_html(p)
        return figures
