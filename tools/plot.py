import argparse
import math
import sys

import islpy as isl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def vertex_to_python(vertex):
    mv = vertex.get_expr().get_constant_multi_val()

    return [mv.get_val(i).to_python() for i in range(len(mv))]

def extract_vertices(domain):
    vertices = []

    domain.compute_vertices().foreach_vertex(
        lambda v: vertices.append(vertex_to_python(v))
    )

    return vertices

def extract_points(domain):
    points = []

    domain.foreach_point(
        lambda pt: points.append(
            [pt.get_coordinate_val(isl.dim_type.set, i).to_python()
                for i in range(2)]
        )
    )

    return points

def plot(points, vertices, dims, out):
    px, py = zip(*points)
    vx, vy = zip(*vertices)

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['svg.fonttype'] = 'none'

    fig, ax = plt.subplots()

    ax.fill(vx, vy, 'lightgrey')
    ax.scatter(px, py, color='black', zorder=2)

    ax.set_aspect('equal')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    ax.set_xlabel(dims[0], loc='right', fontweight='bold')
    ax.set_ylabel(dims[1], loc='top', fontweight='bold', rotation=0)

    ax.grid(color='darkgrey', linestyle='--')
    ax.margins(0.125)

    fig.tight_layout()
    fig.savefig(out, bbox_inches='tight')

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('params', nargs='*', type=int)
    parser.add_argument('-o', '--output', default='domain.svg')

    args = parser.parse_args()

    source = sys.stdin.read()
    domain = isl.BasicSet(source.replace('%', ''))

    for i, param in enumerate(args.params):
        domain = domain.fix_val(isl.dim_type.param, i, param)

    domain = domain.project_out(isl.dim_type.param, 0, len(args.params))
    dims = domain.get_var_names(isl.dim_type.set)

    points = extract_points(domain)
    vertices = extract_vertices(domain)

    cx, cy = (sum(x) / len(x) for x in zip(*vertices))
    vertices.sort(key=lambda pt: math.atan2(pt[1] - cy, pt[0] - cx))

    plot(points, vertices, dims, args.output)

if __name__ == '__main__':
    main()
